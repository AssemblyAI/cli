"""Run logic for `assembly agent-cascade`: the options/run split (see AGENTS.md).

The command module parses argv into an ``AgentCascadeOptions`` and hands it to
``run_agent_cascade``, so tests drive validation and the cascade wiring by
constructing options directly rather than round-tripping through ``CliRunner``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from aai_cli import code_gen
from aai_cli.agent.audio import SAMPLE_RATE, DuplexAudio, NullPlayer
from aai_cli.agent.render import AgentRenderer
from aai_cli.agent_cascade import engine, voices
from aai_cli.agent_cascade.config import DEFAULT_MAX_HISTORY, CascadeConfig
from aai_cli.app.agent_shared import resolve_system_prompt as _resolve_system_prompt
from aai_cli.app.context import AppState
from aai_cli.core import choices, client, config_builder, errors, llm, signals
from aai_cli.core.errors import UsageError
from aai_cli.streaming import turn_presets
from aai_cli.streaming.session import resolve_output_modes
from aai_cli.streaming.sources import FileSource
from aai_cli.tts import session as tts_session
from aai_cli.ui import output

if TYPE_CHECKING:
    from assemblyai.streaming.v3 import StreamingParameters

# A --tts-config key that has its own named flag (or is owned by the cascade), with the
# message steering the user to the right place instead of silently fighting the cascade.
_RESERVED_TTS_KEYS: dict[str, str] = {
    "voice": "Set the voice with --voice, not --tts-config.",
    "language": "Set the language with --language, not --tts-config.",
    "sample_rate": "TTS sample rate is fixed to match the live speaker and can't be overridden.",
}


@dataclass(frozen=True)
class AgentCascadeOptions:
    """Every `assembly agent-cascade` conversation flag as plain data.

    ``--list-voices`` is excluded: it dispatches to its own auth-free body in the
    command module. ``--json`` is excluded: run_command resolves it into the
    ``json_mode`` argument.
    """

    source: str | None
    sample: bool
    voice: str
    model: str
    system_prompt: str
    system_prompt_file: Path | None
    greeting: str
    device: int | None
    output_field: choices.TextOrJson | None
    # Speech-to-text: common knobs named, everything else via --stt-config(-file).
    speech_model: str
    format_turns: bool
    turn_detection: turn_presets.TurnDetectionPreset | None
    stt_config: tuple[str, ...]
    stt_config_file: Path | None
    # Language model: token cap plus any extra gateway request field.
    max_tokens: int
    llm_config: tuple[str, ...]
    # Text-to-speech: language named, any other query param via --tts-config.
    language: str | None
    tts_config: tuple[str, ...]
    # Print the equivalent Python instead of running a conversation.
    show_code: bool


def _build_stt_params(opts: AgentCascadeOptions, sample_rate: int) -> StreamingParameters:
    """Construct the cascade's StreamingParameters from the STT flags + escape hatch.

    A turn-detection preset expands into the three end-of-turn knobs; --stt-config /
    --stt-config-file then override any field (including those knobs). sample_rate is
    fixed by the audio source, so it's merged in here rather than user-set."""
    eot, min_silence, max_silence = turn_presets.resolve(opts.turn_detection, None, None, None)
    flags: dict[str, object] = {
        "speech_model": opts.speech_model,
        "format_turns": opts.format_turns,
        "end_of_turn_confidence_threshold": eot,
        "min_turn_silence": min_silence,
        "max_turn_silence": max_silence,
    }
    merged = config_builder.merge_streaming_params(
        flags=flags | {"sample_rate": sample_rate},
        overrides=opts.stt_config or None,
        config_file=opts.stt_config_file,
    )
    return config_builder.construct_streaming_params(merged)


def _parse_tts_config(pairs: tuple[str, ...]) -> dict[str, str]:
    """Parse --tts-config KEY=VALUE pairs into extra streaming-TTS query params,
    rejecting keys that have a named flag (or are cascade-owned)."""
    extra: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise UsageError(
                f"--tts-config expects KEY=VALUE, got {pair!r}.",
                suggestion="e.g. --tts-config chunk_size_ms=100",
            )
        if key in _RESERVED_TTS_KEYS:
            raise UsageError(_RESERVED_TTS_KEYS[key])
        extra[key] = value
    return extra


def _open_audio(
    renderer: AgentRenderer,
    *,
    source: str | None,
    sample: bool,
    device: int | None,
    from_file: bool,
) -> tuple[Iterable[bytes], engine.Player, int]:
    """Build the (audio, player, sample_rate) triple for file- or mic-driven input."""
    if from_file:
        # Stream the clip as the user's speech; no listener, so discard the reply audio.
        file_source = FileSource(client.resolve_audio_source(source, sample=sample))
        return file_source, NullPlayer(), file_source.sample_rate
    # One full-duplex stream for mic + speaker: macOS rejects two separate streams on
    # one device, which silently kills capture.
    duplex = DuplexAudio(target_rate=SAMPLE_RATE, device=device)
    renderer.notice(
        "Use headphones — the mic stays open while the agent speaks, "
        "so speakers would let it hear itself.\n"
    )
    return duplex.mic, duplex.player, SAMPLE_RATE


def _print_show_code(opts: AgentCascadeOptions, system_prompt_text: str) -> None:
    """Print the equivalent cascade script and exit without authenticating or opening
    audio. Raw stdout for `> script.py`; the named per-leg knobs are reflected, the
    --stt/--llm/--tts-config escape hatches are not."""
    if opts.source or opts.sample:
        # The generated script is microphone-driven (like the agent snippet); a
        # faithful file-driven cascade would need the CLI's ffmpeg-decode +
        # exit-after-reply machinery. Say so on stderr so `--show-code > script.py`
        # stays byte-clean instead of silently dropping the source.
        output.error_console.print(
            "[aai.warn]Note:[/aai.warn] the generated script uses the microphone; "
            "it does not stream the audio source you passed."
        )
    config = CascadeConfig(
        voice=opts.voice,
        system_prompt=system_prompt_text,
        greeting=opts.greeting,
        model=opts.model,
        max_history=DEFAULT_MAX_HISTORY,
        language=opts.language,
        max_tokens=opts.max_tokens,
        format_turns=opts.format_turns,
    )
    output.print_code(code_gen.agent_cascade(config, speech_model=opts.speech_model))


def run_agent_cascade(opts: AgentCascadeOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly agent-cascade` cascade from already-parsed flags."""
    text_mode, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    if opts.voice not in voices.VOICE_NAMES:
        raise UsageError(
            f"Unknown voice {opts.voice!r}.",
            suggestion="Run 'assembly agent-cascade --list-voices' to see the options.",
        )
    # Streaming TTS has no production host, so the whole cascade is sandbox-only.
    tts_session.require_available("agent-cascade")
    system_prompt_text = _resolve_system_prompt(opts.system_prompt, opts.system_prompt_file)

    if opts.show_code:
        _print_show_code(opts, system_prompt_text)
        return

    from_file = bool(opts.source) or opts.sample
    if from_file and opts.device is not None:
        raise UsageError("--device applies only to microphone input.")
    if from_file:
        # Existence-check the clip before credentials, so a typo'd path reads as
        # "file not found" instead of triggering a login.
        client.resolve_audio_source(opts.source, sample=opts.sample)
    # Parse the LLM/TTS escape hatches before opening the device, so a bad KEY=VALUE
    # fails fast instead of after the mic is live.
    llm_extra = llm.parse_gateway_overrides(opts.llm_config)
    tts_extra = _parse_tts_config(opts.tts_config)
    api_key = state.resolve_api_key()

    config = CascadeConfig(
        voice=opts.voice,
        system_prompt=system_prompt_text,
        # File-driven runs speak a clip and end after the reply, so skip the greeting.
        greeting="" if from_file else opts.greeting,
        model=opts.model,
        language=opts.language,
        max_tokens=opts.max_tokens,
        format_turns=opts.format_turns,
        llm_extra=llm_extra,
        tts_extra=tts_extra,
    )
    renderer = AgentRenderer(json_mode=json_mode, text_mode=text_mode, mic_input=not from_file)
    audio, player, sample_rate = _open_audio(
        renderer, source=opts.source, sample=opts.sample, device=opts.device, from_file=from_file
    )
    stt_params = _build_stt_params(opts, sample_rate)
    deps = engine.CascadeDeps.real(api_key, config, audio=audio, stt_params=stt_params)
    try:
        # SIGTERM stops the cascade as cleanly as Ctrl-C, so an external supervisor
        # (Hammerspoon, a service manager, a wrapper's `kill`) can end the session.
        with signals.terminate_as_interrupt():
            engine.run_cascade(renderer=renderer, player=player, config=config, deps=deps)
    except KeyboardInterrupt:
        # Ctrl-C (or a supervisor's SIGTERM) ends the cascade cleanly, then exits 130
        # (cancel) so the interrupt isn't reported to a caller as success.
        renderer.stopped()
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
    except BrokenPipeError as exc:
        # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
        raise typer.Exit(code=0) from exc
    finally:
        with contextlib.suppress(BrokenPipeError):
            renderer.close()
