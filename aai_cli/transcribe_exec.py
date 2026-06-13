"""Transcription execution and result delivery for the ``transcribe`` command.

Kept out of ``commands/transcribe.py`` so the command stays a thin option surface, and
so ``run_transcription`` lives in a core module that ``onboard`` can import directly
(rather than reaching into a command module's internals).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import assemblyai as aai
from rich.markup import escape

from aai_cli import (
    choices,
    client,
    code_gen,
    config_builder,
    jsonshape,
    llm,
    output,
    remotefs,
    stdio,
    transcribe_render,
    transcribe_sources,
    transcribe_validate,
    youtube,
)
from aai_cli.code_gen.transcribe import render as render_transcribe_code
from aai_cli.context import AppState
from aai_cli.errors import UsageError


def render_transform_steps(d: dict[str, Any]) -> str:
    """Human view of chained LLM-Gateway steps: the lone output, or each step labeled."""
    steps = d["transform"]["steps"]
    if len(steps) == 1:
        return str(steps[0]["output"])
    return "\n\n".join(f"Step {i} — {s['prompt']}:\n{s['output']}" for i, s in enumerate(steps, 1))


def out_payload(
    transcript: aai.Transcript,
    output_field: choices.TranscriptOutput | None,
    *,
    chars_per_caption: int | None,
    json_mode: bool,
) -> str:
    """The text to write for ``--out``: the chosen ``-o`` field, the ``--json`` payload,
    or the plain transcript text — the same content stdout would get, as a file artifact."""
    if output_field is not None:
        return client.select_transcript_field(
            transcript, output_field, chars_per_caption=chars_per_caption
        )
    if json_mode:
        return jsonshape.dumps(client.transcript_json_payload(transcript))
    return client.select_transcript_field(transcript, choices.TranscriptOutput.text)


def check_source_exists(source: str | None, *, sample: bool) -> None:
    """Resolve (and existence-check) the audio reference before credential resolution.

    Stdin (``-``) is exempt: its bytes are buffered at transcription time.
    """
    if source != "-":
        client.resolve_audio_source(source, sample=sample, allow_remote=True)


def run_transcription(
    api_key: str,
    source: str | None,
    *,
    sample: bool,
    transcription_config: aai.TranscriptionConfig,
    download_sections: list[str] | None = None,
) -> aai.Transcript:
    if source == "-":
        # Audio piped on stdin (e.g. `ffmpeg -i v.mp4 -f wav - | assembly transcribe -`).
        # The SDK uploads a path, so buffer the bytes to a temp file first.
        data = stdio.read_binary_stdin()
        if not data:
            raise UsageError("No audio received on stdin.")
        with tempfile.TemporaryDirectory(prefix="aai-stdin-") as td:
            local = Path(td) / "audio"
            local.write_bytes(data)
            return client.transcribe(api_key, str(local), config=transcription_config)

    audio = client.resolve_audio_source(source, sample=sample, allow_remote=True)
    if remotefs.is_remote_url(audio):
        # Fetch from bucket/remote storage first; the API can't read s3://-style URLs.
        with tempfile.TemporaryDirectory(prefix="aai-remote-") as td:
            local = remotefs.download(audio, Path(td))
            return client.transcribe(api_key, str(local), config=transcription_config)
    if youtube.is_downloadable_url(audio):
        # Fetch first; AssemblyAI can't read a YouTube/podcast page URL itself.
        with tempfile.TemporaryDirectory(prefix="aai-yt-") as td:
            local = youtube.download_media(audio, Path(td), download_sections=download_sections)
            return client.transcribe(api_key, str(local), config=transcription_config)
    return client.transcribe(api_key, audio, config=transcription_config)


class TransformOptions(NamedTuple):
    """The ``--llm`` chain options: the prompts plus the gateway model settings."""

    prompts: list[str]
    model: str
    max_tokens: int


def deliver_result(
    transcript: aai.Transcript,
    *,
    api_key: str,
    out: Path | None,
    output_field: choices.TranscriptOutput | None,
    chars_per_caption: int | None,
    transform: TransformOptions,
    json_mode: bool,
    quiet: bool,
) -> None:
    """Route the finished transcript: ``--out`` file, single ``-o`` field, ``--llm``
    transform chain, or the default JSON/human render — first match wins."""
    if out is not None:
        # Write a clean file artifact and confirm on stderr; stdout stays empty.
        # The path itself was validated up front by validate_out_path.
        out.write_text(
            out_payload(
                transcript, output_field, chars_per_caption=chars_per_caption, json_mode=json_mode
            )
            + "\n"
        )
        if not quiet:
            output.error_console.print(output.success(f"Saved to {escape(str(out))}"))
        return

    if output_field is not None:
        # Raw single-field output for pipelines (overrides --json and analysis render).
        output.emit_text(
            client.select_transcript_field(
                transcript, output_field, chars_per_caption=chars_per_caption
            )
        )
        return

    if transform.prompts:
        # Chain the prompts: the first runs over the transcript (injected server-side
        # via transcript_id); each subsequent prompt runs over the prior response.
        steps = llm.run_chain_steps(
            api_key,
            transform.prompts,
            transcript_id=transcript.id,
            model=transform.model,
            max_tokens=transform.max_tokens,
        )
        output.emit(
            client.transcript_summary(transcript)
            | {"transform": {"model": transform.model, "steps": steps}},
            render_transform_steps,
            json_mode=json_mode,
        )
        return

    if json_mode:
        output.emit(client.transcript_json_payload(transcript), lambda d: d, json_mode=True)
    else:
        transcribe_render.render_transcript_result(transcript, output.console)


@dataclass(frozen=True)
class TranscribeOptions:
    """Every `assembly transcribe` flag as plain data (options/run split, see AGENTS.md).

    One field per CLI flag (``--json`` excluded: run_command resolves it into the
    ``json_mode`` argument), so a test can describe an invocation without argv.
    """

    source: str | None
    sample: bool
    from_stdin: bool
    concurrency: int
    force: bool
    speech_model: aai.SpeechModel | None
    language_code: str | None
    language_detection: bool | None
    keyterms_prompt: list[str] | None
    temperature: float | None
    prompt: str | None
    punctuate: bool | None
    format_text: bool | None
    disfluencies: bool | None
    speaker_labels: bool
    speakers_expected: int | None
    multichannel: bool | None
    redact_pii: bool | None
    redact_pii_policy: str | None
    redact_pii_sub: aai.PIISubstitutionPolicy | None
    redact_pii_audio: bool | None
    filter_profanity: bool | None
    content_safety: bool | None
    content_safety_confidence: int | None
    speech_threshold: float | None
    summarization: bool | None
    summary_model: aai.SummarizationModel | None
    summary_type: aai.SummarizationType | None
    auto_chapters: bool | None
    sentiment_analysis: bool | None
    entity_detection: bool | None
    auto_highlights: bool | None
    topic_detection: bool | None
    word_boost: list[str] | None
    custom_spelling_file: Path | None
    audio_start: int | None
    audio_end: int | None
    download_sections: list[str] | None
    webhook_url: str | None
    webhook_auth_header: str | None
    translate_to: list[str] | None
    config_kv: list[str] | None
    config_file: Path | None
    llm_prompt: list[str] | None
    model: str
    max_tokens: int
    output_field: choices.TranscriptOutput | None
    chars_per_caption: int | None
    out: Path | None
    show_code: bool

    def flags(self, pii_policies: list[str] | None) -> dict[str, object]:
        """The curated flags in TranscriptionConfig field names (None = unset)."""
        flags: dict[str, object] = {
            "speech_model": config_builder.enum_value(self.speech_model),
            "language_code": self.language_code,
            "language_detection": self.language_detection,
            "keyterms_prompt": list(self.keyterms_prompt) if self.keyterms_prompt else None,
            "temperature": self.temperature,
            "prompt": self.prompt,
            "punctuate": self.punctuate,
            "format_text": self.format_text,
            "disfluencies": self.disfluencies,
            "speaker_labels": self.speaker_labels or None,
            "speakers_expected": self.speakers_expected,
            "multichannel": self.multichannel,
            "redact_pii": self.redact_pii,
            "redact_pii_policies": pii_policies,
            "redact_pii_sub": config_builder.enum_value(self.redact_pii_sub),
            "redact_pii_audio": self.redact_pii_audio,
            "filter_profanity": self.filter_profanity,
            "content_safety": self.content_safety,
            "content_safety_confidence": self.content_safety_confidence,
            "speech_threshold": self.speech_threshold,
            "summarization": self.summarization,
            "summary_model": config_builder.enum_value(self.summary_model),
            "summary_type": config_builder.enum_value(self.summary_type),
            "auto_chapters": self.auto_chapters,
            "sentiment_analysis": self.sentiment_analysis,
            "entity_detection": self.entity_detection,
            "auto_highlights": self.auto_highlights,
            "iab_categories": self.topic_detection,
            "word_boost": list(self.word_boost) if self.word_boost else None,
            "custom_spelling": (
                config_builder.load_custom_spelling(self.custom_spelling_file)
                if self.custom_spelling_file
                else None
            ),
            "audio_start_from": self.audio_start,
            "audio_end_at": self.audio_end,
            "webhook_url": self.webhook_url,
            "speech_understanding": (
                config_builder.translation_request(list(self.translate_to))
                if self.translate_to
                else None
            ),
        }
        flags.update(config_builder.auth_header_flags(self.webhook_auth_header))
        return flags


def _print_show_code(opts: TranscribeOptions, merged: dict[str, object]) -> None:
    """Print the equivalent SDK script and exit without transcribing or authenticating.

    Raw stdout, so `--show-code > script.py` runs. No source/--sample needed — fall
    back to a placeholder path for a pure snippet.
    """
    if opts.source and remotefs.is_remote_url(opts.source):
        raise UsageError(
            "--show-code does not support bucket URLs (s3://, gs://, …) yet.",
            suggestion="Download the audio first and pass the local file.",
        )
    audio = (
        client.resolve_audio_source(opts.source, sample=opts.sample, check_local=False)
        if opts.source or opts.sample
        else "your-audio-file.mp3"
    )
    gateway = code_gen.gateway_options(list(opts.llm_prompt or []), opts.model, opts.max_tokens)
    output.print_code(
        render_transcribe_code(
            merged,
            audio,
            llm_gateway=gateway,
            output=opts.output_field,
            chars_per_caption=opts.chars_per_caption,
            download_sections=list(opts.download_sections or []),
        )
    )


def run_transcribe(opts: TranscribeOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly transcribe` invocation from already-parsed flags."""
    # Module-load order: transcribe_batch imports this module, so import it lazily.
    from aai_cli import transcribe_batch

    transcribe_validate.validate_language_flags(
        opts.language_code, language_detection=opts.language_detection
    )
    pii_policies = config_builder.split_csv(opts.redact_pii_policy)
    transcribe_validate.validate_pii_policies(pii_policies)
    flags = opts.flags(pii_policies)

    transcribe_validate.validate_out_with_llm(opts.out, opts.llm_prompt)
    transcribe_validate.validate_out_path(opts.out)
    transcribe_validate.validate_json_with_output(opts.output_field, json_mode=json_mode)
    client.validate_chars_per_caption(opts.chars_per_caption, opts.output_field)

    merged = config_builder.merge_transcribe_config(
        flags=flags, overrides=opts.config_kv, config_file=opts.config_file
    )
    transcribe_validate.validate_speakers_expected(merged)

    sources = transcribe_sources.expand_sources(
        opts.source, from_stdin=opts.from_stdin, sample=opts.sample
    )
    if sources is not None:
        transcribe_sources.reject_single_source_flags(
            out=opts.out,
            output_field=opts.output_field,
            show_code=opts.show_code,
        )
        transcribe_batch.run_batch(
            state.resolve_api_key(),
            sources,
            transcription_config=config_builder.construct_transcription_config(merged),
            concurrency=opts.concurrency,
            force=opts.force,
            transform=TransformOptions(
                prompts=list(opts.llm_prompt or []), model=opts.model, max_tokens=opts.max_tokens
            ),
            json_mode=json_mode,
            quiet=state.quiet,
        )
        return

    if opts.show_code:
        _print_show_code(opts, merged)
        return

    tc = config_builder.construct_transcription_config(merged)

    # A typo'd path must read as "file not found", not trigger a login.
    check_source_exists(opts.source, sample=opts.sample)
    transcribe_validate.warn_unrecognized_extension(
        opts.source, json_mode=json_mode, quiet=state.quiet
    )

    api_key = state.resolve_api_key()
    with output.status("Transcribing…", json_mode=json_mode, quiet=state.quiet):
        transcript = run_transcription(
            api_key,
            opts.source,
            sample=opts.sample,
            transcription_config=tc,
            download_sections=list(opts.download_sections or []),
        )

    deliver_result(
        transcript,
        api_key=api_key,
        out=opts.out,
        output_field=opts.output_field,
        chars_per_caption=opts.chars_per_caption,
        transform=TransformOptions(
            prompts=list(opts.llm_prompt or []), model=opts.model, max_tokens=opts.max_tokens
        ),
        json_mode=json_mode,
        quiet=state.quiet,
    )
