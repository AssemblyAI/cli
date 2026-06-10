from __future__ import annotations

from pathlib import Path

import assemblyai as aai
import typer
from rich.markup import escape

from aai_cli import (
    choices,
    client,
    code_gen,
    config,
    config_builder,
    help_panels,
    llm,
    options,
    output,
    transcribe_exec,
    transcribe_render,
)

# The package attribute `code_gen.transcribe` is the wrapper function, so the module's
# render() (which also takes the -o output field) is imported from the submodule itself.
from aai_cli.code_gen.transcribe import render as render_transcribe_code
from aai_cli.context import AppState, run_command
from aai_cli.errors import UsageError
from aai_cli.help_text import examples_epilog

app = typer.Typer()

# The PII policy strings the SDK accepts, validated client-side so a typo'd
# --redact-pii-policy fails before any upload — mirroring how an unknown --config
# key is rejected with the valid field list.
_PII_POLICY_VALUES = frozenset(policy.value for policy in aai.PIIRedactionPolicy)


def _validate_pii_policies(policies: list[str] | None) -> None:
    unknown = [p for p in policies or [] if p not in _PII_POLICY_VALUES]
    if unknown:
        valid = ", ".join(sorted(_PII_POLICY_VALUES))
        raise UsageError(f"Unknown PII policy(s) {unknown}. Valid policies: {valid}.")


def _validate_language_flags(language_code: str | None, language_detection: bool | None) -> None:
    if language_code and language_detection:
        raise UsageError(
            "--language-code and --language-detection can't be combined.",
            suggestion="Force a language or auto-detect it, not both.",
        )


def _validate_speakers_expected(merged: dict[str, object]) -> None:
    # Checked on the merged dict so `--config speaker_labels=true` also counts.
    if merged.get("speakers_expected") and not merged.get("speaker_labels"):
        raise UsageError(
            "--speakers-expected only applies when diarization is enabled.",
            suggestion="Add --speaker-labels.",
        )


@app.command(
    rich_help_panel=help_panels.TRANSCRIPTION,
    epilog=examples_epilog(
        [
            ("Transcribe a local file", "aai transcribe call.mp3"),
            ("Try it with the hosted sample", "aai transcribe --sample"),
            ("Transcribe a YouTube video", "aai transcribe https://youtu.be/dtp6b76pMak"),
            ("Label who said what", "aai transcribe call.mp3 --speaker-labels"),
            ("Redact PII for compliance", "aai transcribe call.mp3 --redact-pii"),
            ("Summarize a recording", "aai transcribe call.mp3 --summarization"),
            ("Ask about the transcript", 'aai transcribe call.mp3 --llm "List the action items"'),
        ]
    ),
)
def transcribe(
    ctx: typer.Context,
    source: str | None = typer.Argument(None, help="Audio file path, public URL, or YouTube URL."),
    sample: bool = typer.Option(False, "--sample", help="Use the hosted wildfires.mp3 sample."),
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
        help="Speech model temperature.",
        rich_help_panel=help_panels.OPT_MODEL,
    ),
    prompt: str | None = typer.Option(
        None,
        "--prompt",
        help="Prompt to bias the speech model (u3-pro).",
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
        None, "--audio-end", help="End offset in ms.", rich_help_panel=help_panels.OPT_CUSTOMIZATION
    ),
    # webhooks
    webhook_url: str | None = typer.Option(
        None,
        "--webhook-url",
        help="Webhook URL for completion.",
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
        help="Print one field: text, id, status, utterances, srt (captions), or json.",
    ),
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
    """Transcribe an audio file, URL, or YouTube link.

    Quickest start: aai transcribe call.mp3 (or --sample for the hosted demo). Save with
    --out FILE, or pipe one field with -o text. A YouTube URL is downloaded first, then
    transcribed. Curated flags cover common features; --config KEY=VALUE and --config-file
    reach every other field. Analysis (summary, chapters, ...) renders in human mode.
    """

    def body(state: AppState, json_mode: bool) -> None:
        _validate_language_flags(language_code, language_detection)
        pii_policies = config_builder.split_csv(redact_pii_policy)
        _validate_pii_policies(pii_policies)
        flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(speech_model),
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
            "redact_pii_policies": pii_policies,
            "redact_pii_sub": config_builder.enum_value(redact_pii_sub),
            "redact_pii_audio": redact_pii_audio,
            "filter_profanity": filter_profanity,
            "content_safety": content_safety,
            "content_safety_confidence": content_safety_confidence,
            "speech_threshold": speech_threshold,
            "summarization": summarization,
            "summary_model": config_builder.enum_value(summary_model),
            "summary_type": config_builder.enum_value(summary_type),
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
        flags.update(config_builder.auth_header_flags(webhook_auth_header))

        if out is not None and llm_prompt:
            # --out captures the transcript itself; an LLM transform is a separate step.
            raise UsageError(
                "--out can't be combined with --llm.",
                suggestion='Pipe the transform instead, e.g. -o text | aai llm -f "…".',
            )

        merged = config_builder.merge_transcribe_config(
            flags=flags, overrides=config_kv, config_file=config_file
        )

        _validate_speakers_expected(merged)

        if show_code:
            # Print-only: build the equivalent script and exit without transcribing or
            # authenticating (raw stdout, so `--show-code > script.py` runs). No
            # source/--sample needed — fall back to a placeholder path for a pure snippet.
            audio = (
                client.resolve_audio_source(source, sample=sample, check_local=False)
                if source or sample
                else "your-audio-file.mp3"
            )
            gateway = code_gen.gateway_options(list(llm_prompt or []), model, max_tokens)
            output.print_code(
                render_transcribe_code(merged, audio, llm_gateway=gateway, output=output_field)
            )
            return

        tc = config_builder.construct_transcription_config(merged)

        # A typo'd path must read as "file not found", not trigger a login.
        transcribe_exec.check_source_exists(source, sample=sample)

        api_key = config.resolve_api_key(profile=state.profile)
        with output.status("Transcribing…", json_mode=json_mode):
            transcript = transcribe_exec.run_transcription(
                api_key, source, sample=sample, transcription_config=tc
            )

        if out is not None:
            # Write a clean file artifact and confirm on stderr; stdout stays empty.
            if ".." in out.parts:  # reject path-traversal segments in --out
                raise UsageError(f"--out path can't contain '..': {out}")
            out.write_text(
                transcribe_exec.out_payload(transcript, output_field, json_mode=json_mode) + "\n"
            )
            if not state.quiet:
                output.error_console.print(output.success(f"Saved to {escape(str(out))}"))
            return

        if output_field is not None:
            # Raw single-field output for pipelines (overrides --json and analysis render).
            output.emit_text(client.select_transcript_field(transcript, output_field))
            return

        if llm_prompt:
            # Chain the prompts: the first runs over the transcript (injected server-side
            # via transcript_id); each subsequent prompt runs over the prior response.
            steps = llm.run_chain_steps(
                api_key,
                llm_prompt,
                transcript_id=transcript.id,
                model=model,
                max_tokens=max_tokens,
            )
            output.emit(
                client.transcript_summary(transcript)
                | {"transform": {"model": model, "steps": steps}},
                transcribe_exec.render_transform_steps,
                json_mode=json_mode,
            )
            return

        if json_mode:
            output.emit(client.transcript_json_payload(transcript), lambda d: d, json_mode=True)
        else:
            transcribe_render.render_transcript_result(transcript, output.console)

    run_command(ctx, body, json=json_out)
