from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich.markup import escape

from assemblyai_cli import client, config, llm, output, youtube
from assemblyai_cli.context import AppState, run_command

app = typer.Typer()


def _utterances(transcript: object) -> list[dict[str, object]]:
    """Speaker-labeled utterances ({speaker, text, start, end}), empty if none."""
    items = getattr(transcript, "utterances", None) or []
    return [
        {"speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end} for u in items
    ]


def _render_transcript(data: dict[str, object]) -> str:
    """Human view: speaker-labeled lines when diarized, otherwise the plain text."""
    utterances = data.get("utterances")
    if utterances:
        lines = [f"Speaker {u['speaker']}: {u['text']}" for u in utterances]  # type: ignore[union-attr]
        return escape("\n".join(lines))
    return escape(str(data["text"]))


@app.command()
def transcribe(
    ctx: typer.Context,
    source: str = typer.Argument(None, help="Audio file path, public URL, or YouTube URL."),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
    speaker_labels: bool = typer.Option(False, "--speaker-labels", help="Enable diarization."),
    prompt: str = typer.Option(
        None, "--prompt", help="Bias the speech model with this prompt (u3-pro)."
    ),
    llm_gateway_prompt: str = typer.Option(
        None,
        "--llm-gateway-prompt",
        help="Transform the finished transcript through LLM Gateway with this instruction.",
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL, "--model", help="LLM Gateway model for --llm-gateway-prompt."
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens for the LLM Gateway transform."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Transcribe an audio file, URL, or YouTube URL and print the result.

    --prompt biases the speech model. --llm-gateway-prompt transforms the
    finished transcript through LLM Gateway (e.g. "summarize in three bullets").
    """

    def body(state: AppState, json_mode: bool) -> None:
        audio = client.resolve_audio_source(source, sample=sample)
        api_key = config.resolve_api_key(profile=state.profile)
        if youtube.is_youtube_url(audio):
            # Fetch the audio first; AssemblyAI can't read a YouTube watch URL itself.
            with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
                local = youtube.download_audio(audio, Path(td))
                transcript = client.transcribe(
                    api_key, str(local), speaker_labels=speaker_labels, prompt=prompt
                )
        else:
            transcript = client.transcribe(
                api_key, audio, speaker_labels=speaker_labels, prompt=prompt
            )

        if llm_gateway_prompt:
            transformed = llm.transform_transcript(
                api_key,
                prompt=llm_gateway_prompt,
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
                    "transform": {
                        "model": model,
                        "prompt": llm_gateway_prompt,
                        "output": transformed,
                    },
                },
                lambda d: escape(str(d["transform"]["output"])),
                json_mode=json_mode,
            )
            return

        data: dict[str, object] = {
            "id": transcript.id,
            "status": client.status_str(transcript),
            "text": transcript.text,
        }
        # Surface diarization: --speaker-labels asks for it, so render the per-speaker
        # utterances instead of silently dropping them into the flat .text.
        if speaker_labels:
            utterances = _utterances(transcript)
            if utterances:
                data["utterances"] = utterances
        output.emit(data, _render_transcript, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
