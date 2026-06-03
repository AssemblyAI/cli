from __future__ import annotations

import typer
from rich.markup import escape

from assemblyai_cli import client, config, llm, output
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
    prompt: str = typer.Option(
        None, "--prompt", help="Transform the transcript through LLM Gateway with this instruction."
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model for --prompt."),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens for the --prompt transform."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Transcribe an audio file or URL and print the result.

    Pass --prompt to transform the finished transcript through LLM Gateway
    (e.g. --prompt "summarize in three bullets").
    """

    def body(state: AppState, json_mode: bool) -> None:
        if srt and vtt:
            raise UsageError("--srt and --vtt are mutually exclusive.")
        if prompt and (srt or vtt):
            raise UsageError("--prompt cannot be combined with --srt/--vtt.")
        audio = client.resolve_audio_source(source, sample=sample)
        api_key = config.resolve_api_key(profile=state.profile)
        transcript = client.transcribe(api_key, audio, speaker_labels=speaker_labels)

        # Subtitle formats are inherently plain text; --json does not apply here.
        if srt:
            output.console.print(transcript.export_subtitles_srt(), markup=False)
            return
        if vtt:
            output.console.print(transcript.export_subtitles_vtt(), markup=False)
            return

        if prompt:
            transformed = llm.transform_transcript(
                api_key,
                prompt=prompt,
                model=model,
                transcript_id=transcript.id,
                max_tokens=max_tokens,
            )
            # Human mode shows just the transform; JSON keeps the raw transcript too.
            output.emit(
                {
                    "id": transcript.id,
                    "status": client.status_str(transcript),
                    "text": transcript.text,
                    "transform": {"model": model, "prompt": prompt, "output": transformed},
                },
                lambda d: escape(str(d["transform"]["output"])),
                json_mode=json_mode,
            )
            return

        output.emit(
            {
                "id": transcript.id,
                "status": client.status_str(transcript),
                "text": transcript.text,
            },
            lambda d: escape(str(d["text"])),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
