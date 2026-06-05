from __future__ import annotations

import tempfile
from pathlib import Path

import typer

from aai_cli import (
    client,
    code_gen,
    config,
    config_builder,
    llm,
    output,
    stdio,
    transcribe_render,
    youtube,
)
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.help_text import examples_epilog

app = typer.Typer()


def _render_transform_steps(d: dict) -> str:
    """Human view of chained LLM-Gateway steps: the lone output, or each step labeled."""
    steps = d["transform"]["steps"]
    if len(steps) == 1:
        return str(steps[0]["output"])
    return "\n\n".join(f"Step {i} — {s['prompt']}:\n{s['output']}" for i, s in enumerate(steps, 1))


@app.command(
    epilog=examples_epilog(
        [
            ("Transcribe a local file", "aai transcribe call.mp3"),
            ("Try it with the hosted sample", "aai transcribe --sample"),
            (
                "Diarize two speakers and redact PII",
                "aai transcribe call.mp3 --speaker-labels --speakers-expected 2 --redact-pii",
            ),
            ("Get just the text for a pipeline", "aai transcribe call.mp3 -o text"),
            ("Print equivalent Python instead of running", "aai transcribe call.mp3 --show-code"),
        ]
    )
)
def transcribe(
    ctx: typer.Context,
    source: str = typer.Argument(None, help="Audio file path, public URL, or YouTube URL."),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
    # model & language
    speech_model: str = typer.Option(None, "--speech-model", help="best, nano, slam-1, universal."),
    language_code: str = typer.Option(
        None, "--language-code", help="Force a language (e.g. en_us)."
    ),
    language_detection: bool = typer.Option(
        None, "--language-detection", help="Auto-detect the spoken language."
    ),
    keyterms_prompt: list[str] = typer.Option(
        None, "--keyterms-prompt", help="Boost a key term (repeatable)."
    ),
    temperature: float = typer.Option(None, "--temperature", help="Speech model temperature."),
    prompt: str = typer.Option(None, "--prompt", help="Bias the speech model (u3-pro)."),
    # formatting
    punctuate: bool = typer.Option(None, "--punctuate/--no-punctuate", help="Add punctuation."),
    format_text: bool = typer.Option(None, "--format-text/--no-format-text", help="Format text."),
    disfluencies: bool = typer.Option(None, "--disfluencies", help="Keep filler words."),
    # speakers & channels
    speaker_labels: bool = typer.Option(False, "--speaker-labels", help="Enable diarization."),
    speakers_expected: int = typer.Option(None, "--speakers-expected", help="Hint speaker count."),
    multichannel: bool = typer.Option(None, "--multichannel", help="Transcribe each channel."),
    # guardrails
    redact_pii: bool = typer.Option(None, "--redact-pii", help="Redact PII from the transcript."),
    redact_pii_policy: str = typer.Option(
        None, "--redact-pii-policy", help="Comma-separated PII policies (e.g. person_name,...)."
    ),
    redact_pii_sub: str = typer.Option(
        None, "--redact-pii-sub", help="Substitution: hash or entity_name."
    ),
    redact_pii_audio: bool = typer.Option(None, "--redact-pii-audio", help="Also redact audio."),
    filter_profanity: bool = typer.Option(None, "--filter-profanity", help="Mask profanity."),
    content_safety: bool = typer.Option(None, "--content-safety", help="Detect sensitive content."),
    content_safety_confidence: int = typer.Option(
        None, "--content-safety-confidence", help="Confidence threshold 25-100."
    ),
    speech_threshold: float = typer.Option(
        None, "--speech-threshold", help="Minimum speech proportion 0-1."
    ),
    # analysis
    summarization: bool = typer.Option(None, "--summarization", help="Summarize the transcript."),
    summary_model: str = typer.Option(
        None, "--summary-model", help="informative/conversational/catchy."
    ),
    summary_type: str = typer.Option(
        None, "--summary-type", help="bullets/gist/headline/paragraph."
    ),
    auto_chapters: bool = typer.Option(None, "--auto-chapters", help="Generate chapters."),
    sentiment_analysis: bool = typer.Option(
        None, "--sentiment-analysis", help="Analyze sentiment."
    ),
    entity_detection: bool = typer.Option(None, "--entity-detection", help="Detect entities."),
    auto_highlights: bool = typer.Option(None, "--auto-highlights", help="Detect key phrases."),
    topic_detection: bool = typer.Option(None, "--topic-detection", help="Detect IAB topics."),
    # customization
    word_boost: list[str] = typer.Option(None, "--word-boost", help="Boost a word (repeatable)."),
    custom_spelling_file: str = typer.Option(
        None, "--custom-spelling-file", help="JSON map of custom spellings."
    ),
    audio_start: int = typer.Option(None, "--audio-start", help="Start offset in ms."),
    audio_end: int = typer.Option(None, "--audio-end", help="End offset in ms."),
    # webhooks
    webhook_url: str = typer.Option(None, "--webhook-url", help="Webhook URL for completion."),
    webhook_auth_header: str = typer.Option(
        None, "--webhook-auth-header", help="Webhook auth header as NAME:VALUE."
    ),
    # speech understanding
    translate_to: list[str] = typer.Option(
        None, "--translate-to", help="Translate transcript to a language (repeatable)."
    ),
    # escape hatch
    config_kv: list[str] = typer.Option(
        None, "--config", help="Set any TranscriptionConfig field as KEY=VALUE (repeatable)."
    ),
    config_file: str = typer.Option(None, "--config-file", help="JSON file of config fields."),
    # llm gateway transform
    llm_prompt: list[str] = typer.Option(
        None,
        "--llm",
        help="Transform the finished transcript through LLM Gateway. Repeatable: each "
        "prompt runs on the previous one's response (a chain), the first on the transcript.",
    ),
    model: str = typer.Option(llm.DEFAULT_MODEL, "--model", help="LLM Gateway model."),
    max_tokens: int = typer.Option(llm.DEFAULT_MAX_TOKENS, "--max-tokens", help="Max tokens."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
    output_field: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field of the result: text, id, status, utterances, srt, or json.",
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not transcribe).",
    ),
) -> None:
    """Transcribe an audio file, URL, or YouTube URL with the full TranscriptionConfig surface.

    A YouTube URL is downloaded first, then transcribed. Curated flags cover common
    features; --config KEY=VALUE and --config-file reach every other field. Analysis
    results (summary, chapters, sentiment, ...) render automatically in human mode.
    """

    def body(state: AppState, json_mode: bool) -> None:
        output.validate_output_field(output_field, client.TRANSCRIPT_OUTPUT_FIELDS)
        flags: dict[str, object] = {
            "speech_model": speech_model,
            "language_code": language_code,
            "language_detection": language_detection,
            "keyterms_prompt": list(keyterms_prompt) if keyterms_prompt else None,
            "temperature": temperature,
            "prompt": prompt,
            "punctuate": punctuate,
            "format_text": format_text,
            "disfluencies": disfluencies,
            "speaker_labels": speaker_labels or None,
            "speakers_expected": speakers_expected,
            "multichannel": multichannel,
            "redact_pii": redact_pii,
            "redact_pii_policies": config_builder.split_csv(redact_pii_policy),
            "redact_pii_sub": redact_pii_sub,
            "redact_pii_audio": redact_pii_audio,
            "filter_profanity": filter_profanity,
            "content_safety": content_safety,
            "content_safety_confidence": content_safety_confidence,
            "speech_threshold": speech_threshold,
            "summarization": summarization,
            "summary_model": summary_model,
            "summary_type": summary_type,
            "auto_chapters": auto_chapters,
            "sentiment_analysis": sentiment_analysis,
            "entity_detection": entity_detection,
            "auto_highlights": auto_highlights,
            "iab_categories": topic_detection,
            "word_boost": list(word_boost) if word_boost else None,
            "custom_spelling": (
                config_builder.load_custom_spelling(custom_spelling_file)
                if custom_spelling_file
                else None
            ),
            "audio_start_from": audio_start,
            "audio_end_at": audio_end,
            "webhook_url": webhook_url,
            "speech_understanding": (
                config_builder.translation_request(list(translate_to)) if translate_to else None
            ),
        }
        header = config_builder.parse_auth_header(webhook_auth_header)
        if header is not None:
            flags["webhook_auth_header_name"] = header[0]
            flags["webhook_auth_header_value"] = header[1]

        merged = config_builder.merge_transcribe_config(
            flags=flags, overrides=list(config_kv or []), config_file=config_file
        )

        if show_code:
            # Print-only: build the equivalent script from the flags and exit without
            # transcribing or authenticating. Raw stdout so `--show-code > script.py`
            # yields a runnable file.
            audio = client.resolve_audio_source(source, sample=sample)
            gateway = (
                {"prompts": list(llm_prompt), "model": model, "max_tokens": max_tokens}
                if llm_prompt
                else None
            )
            output.print_code(code_gen.transcribe(merged, audio, llm_gateway=gateway))
            return

        tc = config_builder.construct_transcription_config(merged)

        api_key = config.resolve_api_key(profile=state.profile)
        if source == "-":
            # Audio piped on stdin (e.g. `ffmpeg -i v.mp4 -f wav - | aai transcribe -`).
            # The SDK uploads a path, so buffer the bytes to a temp file first.
            data = stdio.read_binary_stdin()
            if not data:
                raise UsageError("No audio received on stdin.")
            with tempfile.TemporaryDirectory(prefix="aai-stdin-") as td:
                local = Path(td) / "audio"
                local.write_bytes(data)
                transcript = client.transcribe(api_key, str(local), config=tc)
        else:
            audio = client.resolve_audio_source(source, sample=sample)
            if youtube.is_youtube_url(audio):
                # Fetch first; AssemblyAI can't read a YouTube watch URL itself.
                with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
                    local = youtube.download_audio(audio, Path(td))
                    transcript = client.transcribe(api_key, str(local), config=tc)
            else:
                transcript = client.transcribe(api_key, audio, config=tc)

        if output_field is not None:
            # Raw single-field output for pipelines (overrides --json and analysis render).
            print(client.select_transcript_field(transcript, output_field))
            return

        if llm_prompt:
            # Chain the prompts: the first runs over the transcript (injected server-side
            # via transcript_id); each subsequent prompt runs over the prior response.
            steps: list[dict[str, str]] = []
            previous: str | None = None
            for i, prompt_text in enumerate(llm_prompt):
                # First prompt runs over the transcript (by id); each later one over
                # the prior response.
                target = (
                    {"transcript_id": transcript.id} if i == 0 else {"transcript_text": previous}
                )
                out = llm.transform_transcript(
                    api_key, prompt=prompt_text, model=model, max_tokens=max_tokens, **target
                )
                steps.append({"prompt": prompt_text, "output": out})
                previous = out
            output.emit(
                {
                    **client.transcript_summary(transcript),
                    "transform": {"model": model, "steps": steps},
                },
                _render_transform_steps,
                json_mode=json_mode,
            )
            return

        if json_mode:
            output.emit(client.transcript_json_payload(transcript), lambda d: d, json_mode=True)
        else:
            transcribe_render.render_transcript_result(transcript, output.console)

    run_command(ctx, body, json=json_out)
