"""Run logic for `assembly dictate`: the options/run split (see AGENTS.md).

Headless dictation over the Dictation API: recording starts immediately and runs
until SIGTERM is delivered (or the duration cap), then the utterance is POSTed to
the Dictation API, the send-ready text (its cleanup rewrite, or the verbatim
transcript with --verbatim) is printed, and dictate exits 0. There is no terminal
interaction — a controller like Hammerspoon launches `assembly dictate` as a
background task and sends SIGTERM (``task:terminate()`` / ``kill -TERM``) to mean
"I'm done dictating", so the transcript flows straight to the next command in a
pipe. SIGINT (Ctrl-C) cancels without transcribing (exit 130). The command module
(aai_cli/commands/dictate/__init__.py) only parses argv into a ``DictateOptions``;
tests drive the session by constructing options directly and injecting the
stop-signal/mic/HTTP boundaries, with no real signals, microphone, or network.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer

from aai_cli.app.context import AppState
from aai_cli.core import choices, dictation, errors
from aai_cli.core.config_builder import split_csv
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.core.signals import stop_on_terminate
from aai_cli.streaming.validate import resolve_output_modes
from aai_cli.ui import output

# Capture is resampled to one rate, sent as raw 16 kHz mono PCM16 (the live endpoint
# takes WAV or raw PCM; a 120 s utterance is ~3.8 MB, far under the 256 MB cap).
TARGET_RATE = 16000
_BYTES_PER_SECOND = TARGET_RATE * 2  # PCM16 mono


@dataclass(frozen=True)
class DictateOptions:
    """Every `assembly dictate` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    language: str | None
    prompt: str | None
    keyterms: list[str] | None
    device: int | None
    # Deprecated no-op: recording one utterance and exiting is now the default,
    # so --once is kept only so existing scripts don't break (it warns + does
    # nothing). See run_dictate.
    once: bool
    max_seconds: float
    # -o/--output: text (the default bare-transcript shape) or json (== --json).
    output_field: choices.TextOrJson | None = None
    # The rewrite: a custom instruction (replaces the default cleanup), its tier,
    # and whether to print the verbatim transcript instead of the rewrite.
    instruction: str | None = None
    rewrite_mode: choices.RewriteMode | None = None
    verbatim: bool = False
    # Deprecated --word-boost terms (a hidden alias of --keyterm that warns).
    word_boost: list[str] | None = None


def _note(message: str, *, json_mode: bool, quiet: bool) -> None:
    """A muted stderr hint naming how to finish the recording; silent under --json
    (stderr must stay machine-readable) and --quiet."""
    if json_mode or quiet:
        return
    output.error_console.print(f"[aai.muted]{message}[/aai.muted]")


def _config(opts: DictateOptions) -> dictation.DictationConfig:
    """The Dictation config from the flags: --language as ``language_codes`` (one
    code, or a comma-separated list for code-switching audio), --keyterm (plus the
    deprecated --word-boost) as ``keyterms_prompt``, and the rewrite knobs."""
    keyterms = [*(opts.keyterms or []), *(opts.word_boost or [])]
    return dictation.DictationConfig(
        language_codes=split_csv(opts.language) or None,
        stt_prompt=opts.prompt,
        keyterms=keyterms or None,
        llm_instruction=opts.instruction,
        rewrite_mode=opts.rewrite_mode,
    )


def _record(
    stop_requested: Callable[[], bool], mic: MicrophoneSource, *, max_seconds: float
) -> bytes:
    """Capture PCM until SIGTERM is delivered (``stop_requested`` flips True) or the
    duration cap is hit.

    The stop poll runs between ~100 ms mic chunks, so a SIGTERM is honored within one
    chunk without blocking the mic read loop.
    """
    pcm = bytearray()
    frames = iter(mic)
    try:
        for chunk in frames:
            pcm += chunk
            if len(pcm) >= int(max_seconds * _BYTES_PER_SECOND):
                break
            if stop_requested():
                break
    finally:
        # MicrophoneSource yields from a generator whose cleanup releases the
        # device; close it deterministically instead of waiting on GC. Injected
        # fakes (a plain list iterator) may not have close().
        close = getattr(frames, "close", None)
        if callable(close):
            close()
    return bytes(pcm)


def _emit(result: dictation.DictationResult, *, verbatim: bool, json_mode: bool) -> None:
    """One utterance to stdout: the send-ready text (or the verbatim transcript), or
    one NDJSON object carrying both."""
    if json_mode:
        # "type" first: every multi-line NDJSON stream the CLI emits discriminates
        # its lines the same way (stream/agent already do; see docs/cli-reference.md).
        output.emit_ndjson(
            {
                "type": "utterance",
                "text": result.text,
                "llm_response": result.llm_response,
                "llm_error": result.llm_error,
                "confidence": result.confidence,
                "audio_duration_ms": result.audio_duration_ms,
                "session_id": result.session_id,
            }
        )
        return
    if verbatim:
        output.emit_text(result.text)
        return
    if result.llm_error is not None:
        # The rewrite is best-effort and never fails the call: say why the verbatim
        # transcript is printed instead of quietly handing it on.
        output.emit_warning(
            f"The cleanup rewrite failed ({result.llm_error}); printing the verbatim transcript.",
            json_mode=json_mode,
        )
    output.emit_text(result.final_text)


def _transcribe_utterance(
    api_key: str,
    pcm: bytes,
    opts: DictateOptions,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Send one recorded utterance to the Dictation API and print the result.

    An empty recording (an instant SIGTERM) is skipped with a warning rather than
    bounced off the server, which refuses an empty audio part with a 400.
    """
    if not pcm:
        output.emit_warning("Nothing was recorded; nothing to transcribe.", json_mode=json_mode)
        return
    with output.status("Transcribing…", json_mode=json_mode, quiet=state.quiet):
        result = dictation.transcribe_live(
            api_key, pcm, sample_rate=TARGET_RATE, config=_config(opts)
        )
    _emit(result, verbatim=opts.verbatim, json_mode=json_mode)


def _capture_and_transcribe(
    stop_requested: Callable[[], bool],
    api_key: str,
    opts: DictateOptions,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Record one utterance from the mic and print its transcript."""
    mic = MicrophoneSource(
        target_rate=TARGET_RATE,
        device=opts.device,
        on_open=lambda: _note(
            "● Recording — send SIGTERM to transcribe (Ctrl-C cancels).",
            json_mode=json_mode,
            quiet=state.quiet,
        ),
    )
    pcm = _record(stop_requested, mic, max_seconds=opts.max_seconds)
    _transcribe_utterance(api_key, pcm, opts, state, json_mode=json_mode)


def run_dictate(opts: DictateOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly dictate` invocation from already-parsed flags."""
    # Fold -o/--output into json_mode (-o json == --json) and reject the
    # contradictory --json + -o text pair, the same way `stream`/`agent` do.
    # dictate has no live panel, so the text_mode half is unused — plain
    # transcript text is already the non-JSON default in `_emit`.
    _, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    # Resolve credentials before recording: don't capture audio we can't transcribe.
    api_key = state.resolve_api_key()
    if opts.word_boost and not state.quiet:
        # Deprecation trap, not removal: the Dictation API calls these keyterms.
        output.emit_warning("--word-boost is deprecated; use --keyterm.", json_mode=json_mode)
    if opts.once and not state.quiet:
        # Deprecation trap, not removal: --once still parses so old scripts don't
        # break, but recording one utterance and exiting is now the default, so the
        # flag does nothing — say so once (mirrors `login`).
        output.emit_warning(
            "--once is now the default and can be omitted.",
            json_mode=json_mode,
        )
    try:
        # Recording auto-starts and exits after one utterance: SIGTERM stops the
        # capture, which also closes a piped stdout so `assembly dictate | assembly
        # llm …` unblocks the downstream command.
        with stop_on_terminate() as stop_requested:
            _capture_and_transcribe(stop_requested, api_key, opts, state, json_mode=json_mode)
    except KeyboardInterrupt:
        # Ctrl-C / SIGINT cancels dictation, so it exits 130 (cancel) — distinct from
        # SIGTERM, which finishes the utterance normally (exit 0).
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
