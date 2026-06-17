"""Run logic for `assembly dictate`: the options/run split (see AGENTS.md).

Headless dictation over the Sync STT API: recording starts immediately and runs
until SIGTERM is delivered (or the duration cap), then the utterance is POSTed to
the Sync API, the transcript is printed, and dictate exits 0. There is no terminal
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
from aai_cli.core import choices, errors, sync_stt
from aai_cli.core.config_builder import split_csv
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.core.signals import stop_on_terminate
from aai_cli.streaming.validate import resolve_output_modes
from aai_cli.ui import output

# Capture is resampled to one rate the Sync API accepts; 16 kHz mono PCM16 keeps
# a 120 s utterance well under the 40 MB upload cap.
TARGET_RATE = 16000
_BYTES_PER_SECOND = TARGET_RATE * 2  # PCM16 mono


@dataclass(frozen=True)
class DictateOptions:
    """Every `assembly dictate` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    language: str | None
    prompt: str | None
    word_boost: list[str] | None
    device: int | None
    # Deprecated no-op: recording one utterance and exiting is now the default,
    # so --once is kept only so existing scripts don't break (it warns + does
    # nothing). See run_dictate.
    once: bool
    max_seconds: float
    # -o/--output: text (the default bare-transcript shape) or json (== --json).
    output_field: choices.TextOrJson | None = None


def _note(message: str, *, json_mode: bool, quiet: bool) -> None:
    """A muted stderr hint naming how to finish the recording; silent under --json
    (stderr must stay machine-readable) and --quiet."""
    if json_mode or quiet:
        return
    output.error_console.print(f"[aai.muted]{message}[/aai.muted]")


def _languages(language: str | None) -> str | list[str] | None:
    """Fold --language into the config shape: one ISO code as a string, a
    comma-separated list (code-switching audio) as a list, blank as unset."""
    codes = split_csv(language)
    if not codes:
        return None
    return codes[0] if len(codes) == 1 else codes


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


def _emit(result: sync_stt.SyncTranscript, *, json_mode: bool) -> None:
    """One utterance to stdout: the bare transcript text, or one NDJSON object."""
    if json_mode:
        # "type" first: every multi-line NDJSON stream the CLI emits discriminates
        # its lines the same way (stream/agent already do; see docs/cli-reference.md).
        output.emit_ndjson(
            {
                "type": "utterance",
                "text": result.text,
                "confidence": result.confidence,
                "audio_duration_ms": result.audio_duration_ms,
                "session_id": result.session_id,
            }
        )
    else:
        output.emit_text(result.text)


def _transcribe_utterance(
    api_key: str,
    pcm: bytes,
    opts: DictateOptions,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Send one recorded utterance to the Sync API and print the transcript.

    A recording below the API's 80 ms floor (an instant SIGTERM) is skipped with a
    warning rather than bounced off the server as a 400.
    """
    if len(pcm) < sync_stt.MIN_AUDIO_MS * _BYTES_PER_SECOND // 1000:
        output.emit_warning(
            f"Recording was shorter than {sync_stt.MIN_AUDIO_MS} ms; nothing to transcribe.",
            json_mode=json_mode,
        )
        return
    with output.status("Transcribing…", json_mode=json_mode, quiet=state.quiet):
        result = sync_stt.transcribe_pcm(
            api_key,
            pcm,
            sample_rate=TARGET_RATE,
            language_code=_languages(opts.language),
            prompt=opts.prompt,
            word_boost=opts.word_boost,
        )
    _emit(result, json_mode=json_mode)


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
    if opts.prompt and opts.language:
        # The server ignores language_code whenever a custom prompt is set;
        # never drop a requested flag silently (mirrors the speak warnings).
        output.emit_warning(
            "--language is ignored when --prompt is set; state the language inside the prompt.",
            json_mode=json_mode,
        )
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
