"""Run logic for `assembly speak`: the options/run split (see AGENTS.md).

The command module (aai_cli/commands/speak/__init__.py) only parses argv — it builds a
``SpeakOptions`` and hands it to ``run_speak`` via ``context.run_command``, so tests
can drive text resolution, voice assignment, and synthesis wiring by constructing
options directly, with no CliRunner argv round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aai_cli import output, stdio
from aai_cli.context import AppState
from aai_cli.errors import UsageError
from aai_cli.tts import audio, dialogue, session, voices

# The streaming-TTS reference client defaults to English, so the CLI does the
# same. The default voice follows the language (voices.default_voice): each
# voice speaks one language, so e.g. --language Italian switches to giovanni
# unless --voice overrides it.
DEFAULT_LANGUAGE = "English"


@dataclass(frozen=True)
class SpeakOptions:
    """Every `assembly speak` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    text: str | None
    voice: list[str]
    language: str
    sample_rate: int | None
    out: Path | None


def _read_text(text: str | None) -> str:
    """The text to speak: the non-blank argument, or piped stdin when the argument
    is omitted entirely. A *blank* argument (e.g. "") is a usage error, never a
    silent fall-through to stdin — so `assembly speak "$MSG"` with an empty MSG fails
    fast instead of consuming whatever happens to be on the pipe."""
    if text is not None and text.strip():
        return text
    # `text is None` (argument omitted), not merely blank: see the docstring rationale.
    if text is None and (piped := stdio.piped_stdin_text()) is not None:
        return piped.strip()
    raise UsageError(
        "No text to speak.",
        suggestion='Pass text as an argument: assembly speak "Hello" — or pipe it via stdin.',
    )


def _output_audio(result: session.SpeakResult, out: Path | None) -> None:
    """Write a WAV when --out is given, else play through the speakers."""
    if out is not None:
        audio.write_wav(out, result.pcm, result.sample_rate)
    else:
        audio.play_pcm(result.pcm, result.sample_rate)


def _disposition(out: Path | None) -> str:
    return f"saved to {out}" if out is not None else "played"


def _emit(payload: dict[str, object], human_line: str, *, json_mode: bool) -> None:
    """One result summary: the JSON object on stdout, or a muted human note on stderr."""
    if json_mode:
        output.emit_ndjson(payload)
    else:
        output.error_console.print(f"[aai.muted]{human_line}[/aai.muted]")


def _emit_single(
    result: session.SpeakResult,
    cfg: session.SpeakConfig,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    duration = round(result.audio_duration_seconds, 3)
    _emit(
        {
            "voice": cfg.voice,
            "language": cfg.language,
            "sample_rate": result.sample_rate,
            "audio_duration_seconds": duration,
            "bytes": len(result.pcm),
            "out": str(out) if out is not None else None,
        },
        f"Spoke {duration}s of audio ({_disposition(out)}).",
        json_mode=json_mode,
    )


def _emit_multi(
    result: session.SpeakResult,
    speakers: dict[str, str],
    segment_count: int,
    out: Path | None,
    *,
    json_mode: bool,
) -> None:
    duration = round(result.audio_duration_seconds, 3)
    voices = ", ".join(f"{spk}={voice}" for spk, voice in speakers.items())
    _emit(
        {
            "mode": "multi",
            "speakers": speakers,
            "segments": segment_count,
            "sample_rate": result.sample_rate,
            "audio_duration_seconds": duration,
            "bytes": len(result.pcm),
            "out": str(out) if out is not None else None,
        },
        f"Spoke {duration}s across {len(speakers)} voices ({voices}) ({_disposition(out)}).",
        json_mode=json_mode,
    )


def _speak_single(
    api_key: str,
    text: str,
    voice: str,
    opts: SpeakOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    cfg = session.SpeakConfig(
        text=text, voice=voice, language=opts.language, sample_rate=opts.sample_rate
    )
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize(
            api_key, cfg, on_warning=lambda m: output.emit_warning(m, json_mode=json_mode)
        )
    _output_audio(result, opts.out)
    _emit_single(result, cfg, opts.out, json_mode=json_mode)


def _speak_dialogue(
    api_key: str,
    text: str,
    bare_voice: str | None,
    overrides: dict[str, str],
    opts: SpeakOptions,
    *,
    json_mode: bool,
    quiet: bool,
) -> None:
    segments = dialogue.parse_segments(text)
    if not segments:
        raise UsageError(
            "No text to speak.",
            suggestion="The input had speaker labels but no spoken text.",
        )
    if bare_voice is not None:
        output.emit_warning(
            "Ignoring bare --voice in multi-speaker mode; "
            "set a voice per speaker with --voice A=NAME.",
            json_mode=json_mode,
        )
    resolved, speakers = dialogue.assign_voices(
        segments, voices.rotation_for(opts.language), overrides
    )
    with output.status("Synthesizing speech…", json_mode=json_mode, quiet=quiet):
        result = session.synthesize_dialogue(
            api_key,
            resolved,
            language=opts.language,
            sample_rate=opts.sample_rate,
            on_warning=lambda m: output.emit_warning(m, json_mode=json_mode),
        )
    _output_audio(result, opts.out)
    _emit_multi(result, speakers, len(resolved), opts.out, json_mode=json_mode)


def run_speak(opts: SpeakOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly speak` invocation from already-parsed flags."""
    session.require_available("speak")
    spoken = _read_text(opts.text)
    api_key = state.resolve_api_key()
    bare_voice, overrides = dialogue.parse_voice_overrides(opts.voice)
    if dialogue.looks_like_speaker_labeled(spoken):
        _speak_dialogue(
            api_key,
            spoken,
            bare_voice,
            overrides,
            opts,
            json_mode=json_mode,
            quiet=state.quiet,
        )
    else:
        if overrides:
            # Mirror the inverse warning in _speak_dialogue: never drop a
            # requested voice mapping silently.
            output.emit_warning(
                "Ignoring --voice SPEAKER=VOICE mappings; input has no speaker labels.",
                json_mode=json_mode,
            )
        _speak_single(
            api_key,
            spoken,
            bare_voice or voices.default_voice(opts.language),
            opts,
            json_mode=json_mode,
            quiet=state.quiet,
        )
