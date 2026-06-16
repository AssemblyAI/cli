"""Run logic for `assembly agent-framework`: the options/run split (see AGENTS.md).

The command module parses argv into an ``AgentFrameworkOptions`` and hands it to
``run_agent_framework``, so tests drive validation and the cascade wiring by
constructing options directly rather than round-tripping through ``CliRunner``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli.agent.audio import SAMPLE_RATE, DuplexAudio, NullPlayer
from aai_cli.agent.render import AgentRenderer
from aai_cli.agent_framework import engine, voices
from aai_cli.agent_framework.config import CascadeConfig
from aai_cli.app.context import AppState
from aai_cli.core import choices, client
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.streaming.session import resolve_output_modes
from aai_cli.streaming.sources import FileSource
from aai_cli.tts import session as tts_session


@dataclass(frozen=True)
class AgentFrameworkOptions:
    """Every `assembly agent-framework` conversation flag as plain data.

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


def _resolve_system_prompt(system_prompt: str, system_prompt_file: Path | None) -> str:
    """The persona text: a --system-prompt-file (if given) overrides --system-prompt."""
    if system_prompt_file is None:
        return system_prompt
    try:
        return system_prompt_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise CLIError(
            f"Could not read --system-prompt-file {system_prompt_file}: {exc}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Check the path and that the file is readable.",
        ) from exc


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


def run_agent_framework(opts: AgentFrameworkOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly agent-framework` cascade from already-parsed flags."""
    text_mode, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    if opts.voice not in voices.VOICE_NAMES:
        raise UsageError(
            f"Unknown voice {opts.voice!r}.",
            suggestion="Run 'assembly agent-framework --list-voices' to see the options.",
        )
    # Streaming TTS has no production host, so the whole cascade is sandbox-only.
    tts_session.require_available("agent-framework")
    system_prompt_text = _resolve_system_prompt(opts.system_prompt, opts.system_prompt_file)

    from_file = bool(opts.source) or opts.sample
    if from_file and opts.device is not None:
        raise UsageError("--device applies only to microphone input.")
    if from_file:
        # Existence-check the clip before credentials, so a typo'd path reads as
        # "file not found" instead of triggering a login.
        client.resolve_audio_source(opts.source, sample=opts.sample)
    api_key = state.resolve_api_key()

    config = CascadeConfig(
        voice=opts.voice,
        system_prompt=system_prompt_text,
        # File-driven runs speak a clip and end after the reply, so skip the greeting.
        greeting="" if from_file else opts.greeting,
        model=opts.model,
    )
    renderer = AgentRenderer(json_mode=json_mode, text_mode=text_mode, mic_input=not from_file)
    audio, player, sample_rate = _open_audio(
        renderer, source=opts.source, sample=opts.sample, device=opts.device, from_file=from_file
    )
    deps = engine.CascadeDeps.real(api_key, config, audio=audio, sample_rate=sample_rate)
    try:
        engine.run_cascade(renderer=renderer, player=player, config=config, deps=deps)
    except KeyboardInterrupt:
        renderer.stopped()
    except BrokenPipeError as exc:
        # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
        raise typer.Exit(code=0) from exc
    finally:
        with contextlib.suppress(BrokenPipeError):
            renderer.close()
