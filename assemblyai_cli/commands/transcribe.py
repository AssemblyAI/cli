from __future__ import annotations

import typer
from rich.markup import escape

from assemblyai_cli import client, config, output
from assemblyai_cli.context import AppState, run_command
from assemblyai_cli.errors import UsageError

app = typer.Typer()


@app.command()
def transcribe(
    ctx: typer.Context,
    source: str = typer.Argument(None, help="Audio file path or public URL."),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
    speaker_labels: bool = typer.Option(False, "--speaker-labels", help="Enable diarization."),
    srt: bool = typer.Option(False, "--srt", help="Output SRT subtitles."),
    vtt: bool = typer.Option(False, "--vtt", help="Output VTT subtitles."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Transcribe an audio file or URL and print the result."""

    def body(state: AppState, json_mode: bool) -> None:
        if srt and vtt:
            raise UsageError("--srt and --vtt are mutually exclusive.")
        audio = client.SAMPLE_AUDIO_URL if sample else source
        if not audio:
            raise UsageError("Provide an audio path/URL or use --sample.")
        api_key = config.resolve_api_key(profile=state.profile)
        transcript = client.transcribe(api_key, audio, speaker_labels=speaker_labels)

        # Subtitle formats are inherently plain text; --json does not apply here.
        if srt:
            output.console.print(transcript.export_subtitles_srt(), markup=False)
            return
        if vtt:
            output.console.print(transcript.export_subtitles_vtt(), markup=False)
            return

        output.emit(
            {
                "id": transcript.id,
                "status": getattr(transcript.status, "value", transcript.status),
                "text": transcript.text,
            },
            lambda d: escape(str(d["text"])),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
