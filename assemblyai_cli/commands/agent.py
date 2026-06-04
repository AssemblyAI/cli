from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import typer

from assemblyai_cli import client, code_gen, config, output
from assemblyai_cli.agent.audio import SAMPLE_RATE, DuplexAudio, NullPlayer
from assemblyai_cli.agent.render import AgentRenderer
from assemblyai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT, run_session
from assemblyai_cli.agent.voices import DEFAULT_VOICE, VOICES, format_voice_list
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import CLIError, UsageError
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
    system_prompt: str = typer.Option(
        DEFAULT_PROMPT, "--system-prompt", help="System prompt (the agent's persona)."
    ),
    system_prompt_file: Path = typer.Option(
        None,
        "--system-prompt-file",
        help="Read the system prompt from a file (overrides --system-prompt).",
    ),
    greeting: str = typer.Option(DEFAULT_GREETING, "--greeting", help="Spoken greeting."),
    device: int | None = typer.Option(None, "--device", help="Microphone device index."),
    list_voices: bool = typer.Option(False, "--list-voices", help="Print known voices and exit."),
    json_out: bool = typer.Option(False, "--json", help="Emit newline-delimited JSON events."),
    output_field: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Output mode: 'text' (you:/agent: lines as plain stdout, pipe-friendly) or 'json'.",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not start a session).",
    ),
) -> None:
    """Have a live two-way voice conversation with an AssemblyAI voice agent.

    Pass an audio file/URL (or --sample) to speak a recorded clip to the agent
    instead of the microphone; the session then ends after the agent's reply.
    """

    if list_voices:
        typer.echo(format_voice_list())
        raise typer.Exit(code=0)

    def body(state: AppState, json_mode: bool) -> None:
        text_mode, json_mode = output.stream_output_modes(output_field, json_mode)
        if voice not in VOICES:
            raise UsageError(f"Unknown voice {voice!r}. Run 'aai agent --list-voices'.")
        if system_prompt_file is not None:
            try:
                system_prompt_text = system_prompt_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise CLIError(
                    f"Could not read --system-prompt-file {system_prompt_file}: {exc}",
                    error_type="file_not_found",
                    exit_code=2,
                ) from exc
        else:
            system_prompt_text = system_prompt

        if show_code:
            # Print-only: emit the equivalent agent script from the flags and exit
            # without authenticating or opening audio. Raw stdout for `> script.py`.
            output.print_code(code_gen.agent(voice, system_prompt_text, greeting))
            return

        api_key = config.resolve_api_key(profile=state.profile)
        from_file = bool(source) or sample
        if from_file and device is not None:
            raise UsageError("--device applies only to microphone input.")

        renderer = AgentRenderer(
            json_mode=json_mode,
            text_mode=text_mode,
            mic_input=not from_file,
        )
        audio: Any
        player: Any
        if from_file:
            # Stream the clip as the user's speech and stop after the agent replies.
            # No greeting and full-duplex so no part of the clip is muted/dropped,
            # and a NullPlayer since there is no listener for the reply audio.
            audio = FileSource(client.resolve_audio_source(source, sample=sample))
            player = NullPlayer()
        else:
            # One full-duplex stream for mic + speaker: macOS rejects two separate
            # streams on a device, which silently kills capture.
            duplex = DuplexAudio(target_rate=SAMPLE_RATE, device=device)
            audio = duplex.mic
            player = duplex.player
            # notice() self-suppresses in JSON mode and routes to stderr in text mode.
            renderer.notice(
                "Use headphones — the mic stays open while the agent speaks, "
                "so speakers would let it hear itself.\n"
            )
        try:
            run_session(
                api_key,
                renderer=renderer,
                player=player,
                mic=audio,
                voice=voice,
                system_prompt=system_prompt_text,
                greeting="" if from_file else greeting,
                full_duplex=True,  # one duplex stream -> mic always open (use headphones)
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
