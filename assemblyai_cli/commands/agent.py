from __future__ import annotations

import contextlib
from pathlib import Path

import typer

from assemblyai_cli import client, config
from assemblyai_cli.agent.audio import SAMPLE_RATE, NullPlayer, Player
from assemblyai_cli.agent.render import AgentRenderer
from assemblyai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT, run_session
from assemblyai_cli.agent.voices import DEFAULT_VOICE, VOICES, format_voice_list
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import CLIError, UsageError
from assemblyai_cli.microphone import MicrophoneSource
from assemblyai_cli.streaming.sources import FileSource

app = typer.Typer()


@app.command()
def agent(
    ctx: typer.Context,
    source: str = typer.Argument(
        None, help="Audio file path or URL to speak to the agent. Omit to use the microphone."
    ),
    sample: bool = typer.Option(
        False, "--sample", help="Speak the hosted wildfires.mp3 sample to the agent."
    ),
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
    """Have a live two-way voice conversation with an AssemblyAI voice agent.

    Pass an audio file/URL (or --sample) to speak a recorded clip to the agent
    instead of the microphone; the session then ends after the agent's reply.
    """

    if list_voices:
        typer.echo(format_voice_list())
        raise typer.Exit(code=0)

    def body(state: AppState, json_mode: bool) -> None:
        api_key = config.resolve_api_key(profile=state.profile)
        if voice not in VOICES:
            raise UsageError(f"Unknown voice {voice!r}. Run 'aai agent --list-voices'.")
        from_file = bool(source) or sample
        if from_file and device is not None:
            raise UsageError("--device applies only to microphone input.")
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
        audio: FileSource | MicrophoneSource
        player: NullPlayer | Player
        if from_file:
            # Stream the clip as the user's speech and stop after the agent replies.
            # No greeting and full-duplex so no part of the clip is muted/dropped,
            # and a NullPlayer since there is no listener for the reply audio.
            audio = FileSource(client.resolve_audio_source(source, sample=sample))
            player = NullPlayer()
        else:
            audio = MicrophoneSource(sample_rate=SAMPLE_RATE, device=device)
            player = Player(sample_rate=SAMPLE_RATE)
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
                mic=audio,
                voice=voice,
                system_prompt=system_prompt,
                greeting="" if from_file else greeting,
                full_duplex=True if from_file else full_duplex,
                exit_after_reply=from_file,
            )
        except KeyboardInterrupt:
            renderer.stopped()
        except BrokenPipeError as exc:
            # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
            raise typer.Exit(code=0) from exc
        finally:
            with contextlib.suppress(BrokenPipeError):
                renderer.close()

    run_command(ctx, body, json=json_out)
