from __future__ import annotations

from pathlib import Path

import typer

from assemblyai_cli import config
from assemblyai_cli.agent.audio import SAMPLE_RATE, Player
from assemblyai_cli.agent.render import AgentRenderer
from assemblyai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT, run_session
from assemblyai_cli.agent.voices import DEFAULT_VOICE, VOICES, format_voice_list
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import CLIError, UsageError
from assemblyai_cli.microphone import MicrophoneSource

app = typer.Typer()


@app.command()
def agent(
    ctx: typer.Context,
    voice: str = typer.Option(DEFAULT_VOICE, "--voice", help="Agent voice. See --list-voices."),
    prompt: str = typer.Option(DEFAULT_PROMPT, "--prompt", help="System prompt."),
    prompt_file: Path = typer.Option(
        None, "--prompt-file", help="Read the system prompt from a file (overrides --prompt)."
    ),
    greeting: str = typer.Option(DEFAULT_GREETING, "--greeting", help="Spoken greeting."),
    full_duplex: bool = typer.Option(
        False, "--full-duplex", help="Keep the mic open while the agent speaks (needs headphones)."
    ),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    list_voices: bool = typer.Option(False, "--list-voices", help="Print known voices and exit."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
) -> None:
    """Have a live two-way voice conversation with an AssemblyAI voice agent."""

    if list_voices:
        typer.echo(format_voice_list())
        raise typer.Exit(code=0)

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        if voice not in VOICES:
            raise UsageError(f"Unknown voice {voice!r}. Run 'aai agent --list-voices'.")
        system_prompt = prompt
        if prompt_file is not None:
            try:
                system_prompt = prompt_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise CLIError(
                    f"Could not read --prompt-file {prompt_file}: {exc}",
                    error_type="file_not_found",
                    exit_code=2,
                ) from exc

        renderer = AgentRenderer(json_mode=json_mode)
        player = Player(sample_rate=SAMPLE_RATE)
        mic = MicrophoneSource(sample_rate=SAMPLE_RATE, device=device)
        if not json_mode and not full_duplex:
            renderer.notice(
                "Half-duplex: mic mutes while the agent talks. "
                "Use --full-duplex (with headphones) for barge-in.\n"
            )
        try:
            run_session(
                api_key,
                renderer=renderer,
                player=player,
                mic=mic,
                voice=voice,
                system_prompt=system_prompt,
                greeting=greeting,
                full_duplex=full_duplex,
            )
        except KeyboardInterrupt:
            renderer.stopped()
        finally:
            renderer.close()

    run_command(ctx, body, json=json_out)
