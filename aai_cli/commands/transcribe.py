from __future__ import annotations

from pathlib import Path

import assemblyai as aai
import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app import transcribe_exec
from aai_cli.app.context import run_command
from aai_cli.core import choices, llm
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=10,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("transcribe", "t"),  # "t" is the hidden one-letter alias (see below)
)


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Transcribe a local file", "assembly transcribe call.mp3"),
            ("Batch-transcribe a folder", "assembly transcribe ./recordings"),
            ("Batch-transcribe a glob", 'assembly transcribe "calls/*.mp3"'),
            ("Batch-transcribe an S3 prefix", 'assembly transcribe "s3://bucket/calls/*.mp3"'),
            ("Try it with the hosted sample", "assembly transcribe --sample"),
            ("Transcribe a YouTube video", "assembly transcribe https://youtu.be/dtp6b76pMak"),
            ("Transcribe a podcast page", 'assembly transcribe "https://podcasts.apple.com/…"'),
            ("Label who said what", "assembly transcribe call.mp3 --speaker-labels"),
            ("Redact PII for compliance", "assembly transcribe call.mp3 --redact-pii"),
            ("Summarize a recording", "assembly transcribe call.mp3 --summarization"),
            ("Ask about the transcript", 'assembly transcribe call.mp3 --llm "List action items"'),
            ("Summarize a whole folder", 'assembly transcribe ./calls --llm "Summarize this call"'),
        ]
    ),
)
def transcribe(
    ctx: typer.Context,
    source: str | None = typer.Argument(
        None,
        help="Audio file, URL, YouTube/podcast URL, bucket URL (s3://, gs://, …), or a "
        "directory/glob (batch mode).",
    ),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
    # batch mode
    from_stdin: bool = options.batch_from_stdin_option(),
    concurrency: int = options.batch_concurrency_option(),
    force: bool = options.batch_force_option(),
    # model & language
    speech_model: aai.SpeechModel | None = typer.Option(
        None,
        "--speech-model",
        help="Speech model.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    language_code: str | None = typer.Option(
        None,
        "--language-code",
        help="Force a language (e.g. en_us).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    language_detection: bool | None = typer.Option(
        None,
        "--language-detection",
        help="Auto-detect the spoken language.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    keyterms_prompt: list[str] | None = typer.Option(
        None,
        "--keyterms-prompt",
        help="Boost a key term (repeatable).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    temperature: float | None = typer.Option(
        None,
        "--temperature",
        help="Speech model temperature (0 most deterministic, 1 least).",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt to bias the speech model (supported models only).",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    # formatting
    punctuate: bool | None = typer.Option(
        None,
        "--punctuate/--no-punctuate",
        help="Add punctuation.",
        rich_help_panel=help_panels.OPT_FORMATTING,
    ),
    format_text: bool | None = typer.Option(
        None,
        "--format-text/--no-format-text",
        help="Apply text formatting (casing, numbers).",
        rich_help_panel=help_panels.OPT_FORMATTING,
    ),
    disfluencies: bool | None = typer.Option(
        None,
        "--disfluencies",
        help="Keep filler words (e.g. um, uh).",
        rich_help_panel=help_panels.OPT_FORMATTING,
    ),
    # speakers & channels
    speaker_labels: bool = typer.Option(
        False,
        "--speaker-labels",
        help="Enable diarization.",
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    speakers_expected: int | None = typer.Option(
        None,
        "--speakers-expected",
        help="Hint speaker count.",
        min=1,
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    multichannel: bool | None = typer.Option(
        None,
        "--multichannel",
        help="Transcribe each audio channel separately.",
        rich_help_panel=help_panels.OPT_SPEAKERS,
    ),
    # guardrails
    redact_pii: bool | None = typer.Option(
        None,
        "--redact-pii",
        help="Redact PII from the transcript.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_policy: str | None = typer.Option(
        None,
        "--redact-pii-policy",
        help="Comma-separated PII policies (e.g. person_name,...).",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_sub: aai.PIISubstitutionPolicy | None = typer.Option(
        None,
        "--redact-pii-sub",
        help="How to replace redacted PII.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    redact_pii_audio: bool | None = typer.Option(
        None,
        "--redact-pii-audio",
        help="Also redact audio.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    filter_profanity: bool | None = typer.Option(
        None,
        "--filter-profanity",
        help="Mask profanity.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    content_safety: bool | None = typer.Option(
        None,
        "--content-safety",
        help="Detect sensitive content.",
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    content_safety_confidence: int | None = typer.Option(
        None,
        "--content-safety-confidence",
        help="Content-safety confidence threshold (25-100).",
        min=25,
        max=100,
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    speech_threshold: float | None = typer.Option(
        None,
        "--speech-threshold",
        help="Minimum proportion of speech required (0-1).",
        min=0.0,
        max=1.0,
        rich_help_panel=help_panels.OPT_GUARDRAILS,
    ),
    # analysis
    summarization: bool | None = typer.Option(
        None,
        "--summarization",
        help="Summarize the transcript.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    summary_model: aai.SummarizationModel | None = typer.Option(
        None,
        "--summary-model",
        help="Summary model.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    summary_type: aai.SummarizationType | None = typer.Option(
        None,
        "--summary-type",
        help="Summary format.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    auto_chapters: bool | None = typer.Option(
        None, "--auto-chapters", help="Generate chapters.", rich_help_panel=help_panels.OPT_ANALYSIS
    ),
    sentiment_analysis: bool | None = typer.Option(
        None,
        "--sentiment-analysis",
        help="Analyze sentiment.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    entity_detection: bool | None = typer.Option(
        None,
        "--entity-detection",
        help="Detect entities.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    auto_highlights: bool | None = typer.Option(
        None,
        "--auto-highlights",
        help="Detect key phrases.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    topic_detection: bool | None = typer.Option(
        None,
        "--topic-detection",
        help="Detect IAB topics.",
        rich_help_panel=help_panels.OPT_ANALYSIS,
    ),
    # customization
    word_boost: list[str] | None = typer.Option(
        None,
        "--word-boost",
        help="Boost a word (repeatable).",
        rich_help_panel=help_panels.OPT_CUSTOMIZATION,
    ),
    custom_spelling_file: Path | None = typer.Option(
        None,
        "--custom-spelling-file",
        help="JSON map of custom spellings.",
        rich_help_panel=help_panels.OPT_CUSTOMIZATION,
        exists=True,
        dir_okay=False,
    ),
    audio_start: int | None = typer.Option(
        None,
        "--audio-start",
        help="Start offset in ms.",
        min=0,
        rich_help_panel=help_panels.OPT_CUSTOMIZATION,
    ),
    audio_end: int | None = typer.Option(
        None,
        "--audio-end",
        help="End offset in ms.",
        min=0,
        rich_help_panel=help_panels.OPT_CUSTOMIZATION,
    ),
    download_sections: list[str] | None = typer.Option(
        None,
        "--download-sections",
        help="For a YouTube/podcast URL, download only part of the source (yt-dlp "
        '"--download-sections" syntax, e.g. "*0:00-5:00" for the first five minutes; '
        "repeatable).",
        rich_help_panel=help_panels.OPT_CUSTOMIZATION,
    ),
    # webhooks
    webhook_url: str | None = typer.Option(
        None,
        "--webhook-url",
        help="Webhook URL for completion (get a dev URL with `assembly webhooks listen`).",
        rich_help_panel=help_panels.OPT_WEBHOOKS,
    ),
    webhook_auth_header: str | None = typer.Option(
        None,
        "--webhook-auth-header",
        help="Webhook auth header as NAME:VALUE.",
        rich_help_panel=help_panels.OPT_WEBHOOKS,
        metavar="NAME:VALUE",
    ),
    # speech understanding
    translate_to: list[str] | None = typer.Option(
        None,
        "--translate-to",
        help="Translate transcript to a language (repeatable).",
        rich_help_panel=help_panels.OPT_TRANSLATION,
    ),
    # escape hatch
    config_kv: list[str] | None = typer.Option(
        None,
        "--config",
        help="Set any TranscriptionConfig field as KEY=VALUE (repeatable).",
        rich_help_panel=help_panels.OPT_ADVANCED,
        metavar="KEY=VALUE",
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config-file",
        help="JSON file of config fields.",
        rich_help_panel=help_panels.OPT_ADVANCED,
        exists=True,
        dir_okay=False,
    ),
    # llm gateway transform
    llm_prompt: list[str] | None = typer.Option(
        None,
        "--llm",
        help="Transform the finished transcript through LLM Gateway. Repeatable: each "
        "prompt runs on the previous one's response (a chain), the first on the transcript.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    model: str = typer.Option(
        llm.DEFAULT_MODEL,
        "--model",
        help="LLM Gateway model.",
        rich_help_panel=help_panels.OPT_LLM,
        autocompletion=llm.complete_model,
    ),
    max_tokens: int = typer.Option(
        llm.DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max tokens.",
        rich_help_panel=help_panels.OPT_LLM,
    ),
    json_out: bool = options.json_option(
        "Output the full result as JSON. Text stays the default even when piped; "
        "opt in here (same as -o json)."
    ),
    output_field: choices.TranscriptOutput | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Print one field: text, id, status, utterances, srt or vtt (captions), or json.",
    ),
    chars_per_caption: int | None = options.chars_per_caption_option(),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Save the result to a file instead of printing it (clean text; pairs with -o).",
        dir_okay=False,
    ),
    show_code: bool = typer.Option(
        False,
        "--show-code",
        help="Print the equivalent Python SDK code and exit (does not transcribe).",
    ),
) -> None:
    """Transcribe an audio file, URL, or YouTube/podcast link — or a whole batch.

    Quickest start: assembly transcribe call.mp3 (or --sample for the hosted demo).

    Save with --out FILE, or pipe one field with -o text. YouTube and podcast-page
    URLs (any page yt-dlp can extract) are downloaded first, then transcribed.

    Batch mode: pass a directory or glob (or pipe a list with --from-stdin) to
    transcribe many sources concurrently. Each source gets a .aai.json sidecar
    with the full result (including any --llm responses), and a re-run skips
    sources already transcribed — with changed --llm prompts it replays just
    the LLM step, never a second transcription.

    Bucket URLs (s3://, gs://, az://, sftp://, …) work for single files and for
    batches (a glob, or a folder ending in /); install the matching fsspec
    backend first (e.g. pip install s3fs) and use its usual credentials.

    Curated flags cover common features; --config KEY=VALUE and --config-file reach every other field. Analysis (summary, chapters, ...) renders in human mode.
    """
    opts = transcribe_exec.TranscribeOptions(
        source=source,
        sample=sample,
        from_stdin=from_stdin,
        concurrency=concurrency,
        force=force,
        speech_model=speech_model,
        language_code=language_code,
        language_detection=language_detection,
        keyterms_prompt=keyterms_prompt,
        temperature=temperature,
        prompt=prompt,
        punctuate=punctuate,
        format_text=format_text,
        disfluencies=disfluencies,
        speaker_labels=speaker_labels,
        speakers_expected=speakers_expected,
        multichannel=multichannel,
        redact_pii=redact_pii,
        redact_pii_policy=redact_pii_policy,
        redact_pii_sub=redact_pii_sub,
        redact_pii_audio=redact_pii_audio,
        filter_profanity=filter_profanity,
        content_safety=content_safety,
        content_safety_confidence=content_safety_confidence,
        speech_threshold=speech_threshold,
        summarization=summarization,
        summary_model=summary_model,
        summary_type=summary_type,
        auto_chapters=auto_chapters,
        sentiment_analysis=sentiment_analysis,
        entity_detection=entity_detection,
        auto_highlights=auto_highlights,
        topic_detection=topic_detection,
        word_boost=word_boost,
        custom_spelling_file=custom_spelling_file,
        audio_start=audio_start,
        audio_end=audio_end,
        download_sections=download_sections,
        webhook_url=webhook_url,
        webhook_auth_header=webhook_auth_header,
        translate_to=translate_to,
        config_kv=config_kv,
        config_file=config_file,
        llm_prompt=llm_prompt,
        model=model,
        max_tokens=max_tokens,
        output_field=output_field,
        chars_per_caption=chars_per_caption,
        out=out,
        show_code=show_code,
    )
    run_command(
        ctx,
        lambda state, json_mode: transcribe_exec.run_transcribe(opts, state, json_mode=json_mode),
        json=json_out,
    )


# `assembly t` — a one-letter alias for the CLI's highest-frequency command (the
# pattern codex uses for `e`/`a`). Registered hidden so the root help table keeps
# one row per command; `assembly t --help` still renders the full transcribe help.
app.command(
    name="t",
    hidden=True,
    epilog=examples_epilog([("Same flags as transcribe", "assembly t call.mp3 --speaker-labels")]),
)(transcribe)
