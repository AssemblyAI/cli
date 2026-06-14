"""Run logic for `assembly dictate`: the options/run split (see AGENTS.md).

Push-to-talk dictation over the Sync STT API: wait for a hotkey, record the
microphone until the hotkey is pressed again (or the duration cap), POST the
utterance to the Sync API, print the transcript, repeat. The command module
(aai_cli/commands/dictate/__init__.py) only parses argv into a ``DictateOptions``; tests
drive the session by constructing options directly and injecting the key/mic/
HTTP boundaries, with no CliRunner argv round-trip and no real terminal.
"""

from __future__ import annotations

from dataclasses import dataclass

from aai_cli.app.context import AppState
from aai_cli.core import sync_stt
from aai_cli.core.config_builder import split_csv
from aai_cli.core.hotkey import CTRL_C, CTRL_D, ESC, TerminalKeys
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.ui import output

# Capture is resampled to one rate the Sync API accepts; 16 kHz mono PCM16 keeps
# a 120 s utterance well under the 40 MB upload cap.
TARGET_RATE = 16000
_BYTES_PER_SECOND = TARGET_RATE * 2  # PCM16 mono

# Enter or Space toggles recording; q / Esc / Ctrl-D ends the session at the
# idle prompt (Ctrl-C works anywhere — cbreak mode keeps SIGINT delivery).
TOGGLE_KEYS = frozenset({"\r", "\n", " "})
QUIT_KEYS = frozenset({"q", "Q", ESC, CTRL_C, CTRL_D})


@dataclass(frozen=True)
class DictateOptions:
    """Every `assembly dictate` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    language: str | None
    prompt: str | None
    word_boost: list[str] | None
    device: int | None
    once: bool
    max_seconds: float


def _note(message: str, *, json_mode: bool, quiet: bool) -> None:
    """A muted stderr hint guiding the interactive session; silent under --json
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


def _record(keys: TerminalKeys, mic: MicrophoneSource, *, max_seconds: float) -> bytes:
    """Capture PCM until a hotkey is pressed again or the duration cap is hit.

    The key poll runs between ~100 ms mic chunks with a zero timeout, so the mic
    read loop is never blocked waiting on the keyboard.
    """
    pcm = bytearray()
    frames = iter(mic)
    try:
        for chunk in frames:
            pcm += chunk
            if len(pcm) >= int(max_seconds * _BYTES_PER_SECOND):
                break
            # None (no key pending) is simply not in either set.
            if keys.read(0) in TOGGLE_KEYS | QUIT_KEYS:
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

    A recording below the API's 80 ms floor (a double-tapped hotkey) is skipped
    with a warning rather than bounced off the server as a 400.
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


def _session(
    keys: TerminalKeys,
    api_key: str,
    opts: DictateOptions,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """The dictation loop: idle until a toggle key, record, transcribe, repeat."""
    while True:
        key = keys.read(None)
        if key is None or key in QUIT_KEYS:
            return
        if key not in TOGGLE_KEYS:
            continue
        mic = MicrophoneSource(
            target_rate=TARGET_RATE,
            device=opts.device,
            on_open=lambda: _note(
                "● Recording — press Enter to stop.", json_mode=json_mode, quiet=state.quiet
            ),
        )
        pcm = _record(keys, mic, max_seconds=opts.max_seconds)
        _transcribe_utterance(api_key, pcm, opts, state, json_mode=json_mode)
        if opts.once:
            return


def run_dictate(opts: DictateOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly dictate` invocation from already-parsed flags."""
    try:
        # Entering TerminalKeys validates the terminal (a usage precondition)
        # before credentials, so a piped stdin reads as "needs a terminal" — not
        # as a login prompt.
        with TerminalKeys() as keys:
            api_key = state.resolve_api_key()
            if opts.prompt and opts.language:
                # The server ignores language_code whenever a custom prompt is set;
                # never drop a requested flag silently (mirrors the speak warnings).
                output.emit_warning(
                    "--language is ignored when --prompt is set; "
                    "state the language inside the prompt.",
                    json_mode=json_mode,
                )
            _note(
                "Press Enter to start recording, Enter again to transcribe. q quits.",
                json_mode=json_mode,
                quiet=state.quiet,
            )
            _session(keys, api_key, opts, state, json_mode=json_mode)
    except KeyboardInterrupt:
        # Ctrl-C is the normal "done dictating" signal: end cleanly, not as an error.
        return
