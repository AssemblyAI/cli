from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT
from aai_cli.agent.voices import DEFAULT_VOICE, VOICES, complete_voice, format_voice_list
from aai_cli.app.context import AppState, run_command
from aai_cli.commands.agent import _exec as agent_exec
from aai_cli.core import choices
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=40,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("agent",),
)


def _emit_voice_list(_state: AppState, json_mode: bool) -> None:
    """--list-voices body, routed through run_command so --json yields a
    machine-readable array instead of the human list; needs no auth."""
    payload = [{"name": voice.name, "language": voice.language} for voice in VOICES]
    output.emit(payload, lambda _voices: format_voice_list(), json_mode=json_mode)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Start a live voice conversation", "assembly agent"),
            ("Pick a voice and opening line", 'assembly agent --voice james --greeting "Hi there"'),
            (
                "Give the agent a persona",
                'assembly agent --system-prompt "You are a terse pirate."',
            ),
            ("See available voices", "assembly agent --list-voices"),
            ("Print equivalent Python instead of running", "assembly agent --show-code"),
        ]
    ),
)
def agent(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None, help="Audio file path or URL to speak to the agent. Omit to use the microphone."
    ),
    sample: bool = typer.Option(
        False, "--sample", help="Speak the hosted wildfires.mp3 sample to the agent"
    ),
    voice: str = typer.Option(
        DEFAULT_VOICE,
        "--voice",
        help="Agent voice. See --list-voices.",
        autocompletion=complete_voice,
    ),
    system_prompt: str = typer.Option(
        DEFAULT_PROMPT, "--system-prompt", help="System prompt (the agent's persona)"
    ),
    system_prompt_file: Path | None = typer.Option(
        None,
        "--system-prompt-file",
        help="Read the system prompt from a file (overrides --system-prompt)",
        exists=True,
        dir_okay=False,
    ),
    greeting: str = typer.Option(DEFAULT_GREETING, "--greeting", help="Spoken greeting"),
    device: int | None = typer.Option(None, "--device", help="Microphone device index"),
    list_voices: bool = typer.Option(False, "--list-voices", help="Print known voices and exit"),
    json_out: bool = options.json_option("Emit newline-delimited JSON events"),
    output_field: choices.TextOrJson | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: text (you:/agent: lines as plain stdout, pipe-friendly) or json",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not start a session)",
    ),
) -> None:
    """Hold a live two-way voice conversation with a voice agent

    Use headphones: the mic stays open while the agent speaks, so on
    speakers it would hear itself and loop. Pass an audio file/URL (or
    --sample) to speak a recorded clip to the agent instead of the
    microphone; the session then ends after the agent's reply.

    This only runs a conversation in the terminal — it writes no code. To
    build a voice agent app, run 'assembly init voice-agent' instead.
    """

    if list_voices:
        run_command(ctx, _emit_voice_list, json=json_out)
        return

    opts = agent_exec.AgentOptions(
        source=source,
        sample=sample,
        voice=voice,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
        greeting=greeting,
        device=device,
        output_field=output_field,
        show_code=show_code,
    )
    run_command(
        ctx,
        lambda state, json_mode: agent_exec.run_agent(opts, state, json_mode=json_mode),
        json=json_out,
    )
