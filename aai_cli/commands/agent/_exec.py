"""Run logic for `assembly agent`: the options/run split (see AGENTS.md).

The command module (aai_cli/commands/agent/__init__.py) only parses argv — it builds an
``AgentOptions`` and hands it to ``run_agent`` via ``context.run_command``, so tests
can drive validation, --show-code, and session wiring by constructing options
directly, with no CliRunner argv round-trip.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from aai_cli import code_gen
from aai_cli.agent.audio import SAMPLE_RATE, DuplexAudio, NullPlayer
from aai_cli.agent.render import AgentRenderer
from aai_cli.agent.session import AgentRunConfig, run_session
from aai_cli.agent.voices import VOICE_NAMES
from aai_cli.app.agent_shared import resolve_system_prompt as _resolve_system_prompt
from aai_cli.app.agent_shared import validate_voice
from aai_cli.app.context import AppState
from aai_cli.core import choices, client, errors, signals
from aai_cli.core.errors import UsageError
from aai_cli.streaming.sources import FileSource
from aai_cli.streaming.validate import resolve_output_modes
from aai_cli.ui import output


@dataclass(frozen=True)
class AgentOptions:
    """Every `assembly agent` conversation flag as plain data.

    ``--list-voices`` is excluded: it dispatches to its own auth-free body in the
    command module. ``--json`` is excluded: run_command resolves it into the
    ``json_mode`` argument.
    """

    source: str | None
    sample: bool
    voice: str
    system_prompt: str
    system_prompt_file: Path | None
    greeting: str
    device: int | None
    output_field: choices.TextOrJson | None
    show_code: bool


def _open_audio(
    renderer: AgentRenderer,
    *,
    source: str | None,
    sample: bool,
    device: int | None,
    from_file: bool,
) -> tuple[Any, Any]:
    """Build the (mic, player) pair for either file-driven or live-mic input."""
    if from_file:
        # Stream the clip as the user's speech and stop after the agent replies.
        # No greeting and full-duplex so no part of the clip is muted/dropped,
        # and a NullPlayer since there is no listener for the reply audio.
        return FileSource(client.resolve_audio_source(source, sample=sample)), NullPlayer()
    # One full-duplex stream for mic + speaker: macOS rejects two separate
    # streams on a device, which silently kills capture.
    duplex = DuplexAudio(target_rate=SAMPLE_RATE, device=device)
    # notice() self-suppresses in JSON mode and routes to stderr otherwise, so a
    # piped `assembly agent | …` never reads this advisory as transcript data.
    renderer.notice(
        "Use headphones — the mic stays open while the agent speaks, "
        "so speakers would let it hear itself.\n"
    )
    return duplex.mic, duplex.player


def _print_show_code(opts: AgentOptions, system_prompt_text: str) -> None:
    """Print the equivalent agent script and exit without authenticating or opening
    audio. Raw stdout for `> script.py`."""
    if opts.source or opts.sample:
        # A faithful file-driven agent script would need the CLI's whole
        # ffmpeg-decode + ready-gate + exit-after-reply machinery, which is
        # impractical to inline; the snippet is microphone-driven, so say so
        # on stderr instead of silently dropping the source. stderr keeps
        # `--show-code > script.py` byte-clean.
        output.error_console.print(
            "[aai.warn]Note:[/aai.warn] the generated script uses the microphone; "
            "it does not stream the audio source you passed."
        )
    output.print_code(code_gen.agent(opts.voice, system_prompt_text, opts.greeting))


def run_agent(opts: AgentOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly agent` conversation from already-parsed flags."""
    text_mode, json_mode = resolve_output_modes(opts.output_field, json_mode=json_mode)
    validate_voice(opts.voice, VOICE_NAMES, command="agent")
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
    api_key = state.resolve_api_key()

    renderer = AgentRenderer(
        json_mode=json_mode,
        text_mode=text_mode,
        mic_input=not from_file,
    )
    audio, player = _open_audio(
        renderer, source=opts.source, sample=opts.sample, device=opts.device, from_file=from_file
    )
    run_config = AgentRunConfig(
        voice=opts.voice,
        system_prompt=system_prompt_text,
        greeting="" if from_file else opts.greeting,
        full_duplex=True,  # one duplex stream -> mic always open (use headphones)
        exit_after_reply=from_file,
    )
    try:
        # SIGTERM stops the agent as cleanly as Ctrl-C, so an external supervisor
        # (Hammerspoon, a service manager, a wrapper's `kill`) can end the session.
        with signals.terminate_as_interrupt():
            run_session(api_key, renderer=renderer, player=player, mic=audio, config=run_config)
    except KeyboardInterrupt:
        # Ctrl-C (or a supervisor's SIGTERM) ends the session cleanly, then exits 130
        # (cancel) so the interrupt isn't reported to a caller as success.
        renderer.stopped()
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
    except BrokenPipeError as exc:
        # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
        raise typer.Exit(code=0) from exc
    finally:
        with contextlib.suppress(BrokenPipeError):
            renderer.close()
