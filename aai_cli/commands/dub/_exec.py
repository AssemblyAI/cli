"""Run logic for `assembly dub`: transcribe → translate → synthesize → ffmpeg track-swap.

The command module (aai_cli/commands/dub/__init__.py) only parses argv — it builds a
``DubOptions`` and hands it to ``run_dub`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the whole pipeline by
constructing options directly.

The pipeline runs the platform end to end in one command: the media is
transcribed with diarized utterance timestamps, each utterance is translated to
the target language by an LLM Gateway model, each translation is synthesized
with streaming TTS (one voice per speaker), the segments are laid out on a
silence timeline at their original start times, and ffmpeg swaps the new track
over the original media (video stream copied untouched). A YouTube/media-page
URL is downloaded first — audio only, or the full video with ``--video`` so the
dub keeps the picture. Streaming TTS only exists in the sandbox today, so —
like `assembly speak` — the command is sandbox-only.
"""

from __future__ import annotations

import dataclasses
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from aai_cli.app import batch, mediafile
from aai_cli.app.context import AppState
from aai_cli.commands.dub import _pipeline as pipeline
from aai_cli.core import youtube
from aai_cli.core.errors import UsageError, mutually_exclusive
from aai_cli.tts import audio, dialogue, session
from aai_cli.ui import output

# ISO-639-1 codes accepted by --lang, mapped to the language *name* both the
# translation prompt and the streaming-TTS `language` param expect. A value not
# listed passes through as typed, so a full name ("German") — or an unlisted
# language the gateway can translate to — still works.
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


@dataclass(frozen=True)
class DubOptions:
    """Every `assembly dub` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    # Empty in batch mode, where the sources arrive on stdin (--from-stdin).
    media: str
    language: str
    source_language: str | None
    transcript_id: str | None
    voice: list[str]
    model: str
    max_tokens: int
    out: Path | None
    video: bool
    download_sections: list[str]
    from_stdin: bool
    concurrency: int
    force: bool


def resolve_language(value: str) -> str:
    """The target language name: an ISO code maps to its name, anything else
    passes through as typed (the gateway accepts more languages than the map)."""
    cleaned = value.strip()
    if not cleaned:
        raise UsageError(
            "--lang needs a language.",
            suggestion="Pass an ISO code (--lang de) or a language name (--lang German).",
        )
    return LANGUAGE_NAMES.get(cleaned.casefold(), cleaned)


def default_out_path(media: Path, language: str) -> Path:
    """The default output file: ``<stem>.dub.<lang><ext>`` next to the input."""
    slug = re.sub(r"[^a-z0-9]+", "-", language.casefold()).strip("-")
    if not slug:
        # A name that slugs to nothing (e.g. 中文) would collide every such
        # language onto one "<stem>.dub.<ext>" file; make the user pick.
        raise UsageError(
            f"Can't derive a default output name for {language!r}.",
            suggestion="Pass --out explicitly, e.g. --out dubbed.mp4.",
        )
    return media.parent / f"{media.stem}.dub.{slug}{media.suffix}"


def run_dub(opts: DubOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute `assembly dub`: one source, or a stdin batch (`--from-stdin`)."""
    language = resolve_language(opts.language)
    session.require_available("dub")
    # Parse --voice now: a malformed mapping must fail before the billed pipeline.
    voice_plan = pipeline.VoicePlan(*dialogue.parse_voice_overrides(opts.voice))
    sources = batch.stdin_sources(opts.media, from_stdin=opts.from_stdin)
    if sources is not None:
        _reject_batch_conflicts(opts)
        batch.run_batch(
            sources,
            worker=_dub_worker(
                opts, state, language, voice_plan, force=opts.force, json_mode=json_mode
            ),
            concurrency=opts.concurrency,
            summary_verb="Dubbed",
            json_mode=json_mode,
            quiet=state.quiet,
        )
        return
    if not opts.media:
        raise UsageError(
            "Pass a video/audio file or URL to dub, or --from-stdin to read a list from stdin.",
            suggestion="e.g. assembly --sandbox dub talk.mp4 -l de",
        )
    result = _dub_one(opts, state, language, voice_plan, json_mode=json_mode)
    output.emit(
        result.payload, lambda _: output.success(escape(result.summary)), json_mode=json_mode
    )


def _reject_batch_conflicts(opts: DubOptions) -> None:
    """Single-result flags that can't span a many-source batch."""
    mutually_exclusive(
        ("--out", opts.out),
        ("--from-stdin", True),
        suggestion="In batch mode each source gets its own <name>.dub.<lang><ext>; drop --out.",
    )
    mutually_exclusive(
        ("--transcript-id/-t", opts.transcript_id),
        ("--from-stdin", True),
        suggestion="A transcript id can't apply to many sources; drop -t in batch mode.",
    )


def _dub_worker(
    opts: DubOptions,
    state: AppState,
    language: str,
    voice_plan: pipeline.VoicePlan,
    *,
    force: bool,
    json_mode: bool,
) -> batch.Worker:
    """A per-source worker for the batch runner: skip a source whose default output
    already exists (unless ``--force``), else dub it with spinners silenced."""
    quiet_state = dataclasses.replace(state, quiet=True)

    def worker(source: str) -> batch.SourceResult:
        if not force and (existing := _existing_output(source, language)) is not None:
            return batch.SourceResult(
                payload={"source": source, "out": str(existing)},
                summary=f"{existing} exists",
                status="skipped",
            )
        return _dub_one(
            dataclasses.replace(opts, media=source),
            quiet_state,
            language,
            voice_plan,
            json_mode=json_mode,
        )

    return worker


def _existing_output(source: str, language: str) -> Path | None:
    """The default output for a local ``source`` when it already exists (so batch mode
    skips it), else ``None`` — a URL or a source with no prior output, both processed."""
    if "://" in source:
        return None
    out = default_out_path(Path(source), language)
    return out if out.exists() else None


def _dub_one(
    opts: DubOptions,
    state: AppState,
    language: str,
    voice_plan: pipeline.VoicePlan,
    *,
    json_mode: bool,
) -> batch.SourceResult:
    """Resolve ``opts.media`` to a local file, dub it, and return the result.

    A media-page URL is downloaded once — the audio track by default, the full
    video with --video so the dub keeps the picture, only the --download-sections
    slices when given — and dubbed locally.
    """
    youtube.validate_video_flag(opts.media, video=opts.video)
    youtube.validate_sections_flag(opts.media, opts.download_sections)
    # ffmpeg is checked before any (billed) download/transcription so a missing
    # dependency fails before any fetch.
    ffmpeg = mediafile.require_ffmpeg("write the dubbed file")
    with mediafile.resolve_media_source(
        opts.media,
        "dub",
        fetch_clause="dubs a local file or a media-page URL yt-dlp can download (YouTube, podcasts, …)",
        download_suggestion="Download the media first, then dub the local copy.",
        video=opts.video,
        download_sections=opts.download_sections,
        json_mode=json_mode,
        quiet=state.quiet,
    ) as (media, downloaded):
        if not downloaded:
            mediafile.validate_local_media(media, "dub")
        out = mediafile.default_output(
            opts.out, media, downloaded=downloaded, namer=lambda m: default_out_path(m, language)
        )
        mediafile.validate_out(out, media)
        return _dub_build(
            opts, media, out, language, ffmpeg, voice_plan, state, json_mode=json_mode
        )


def _dub_build(
    opts: DubOptions,
    media: Path,
    out: Path,
    language: str,
    ffmpeg: str,
    voice_plan: pipeline.VoicePlan,
    state: AppState,
    *,
    json_mode: bool,
) -> batch.SourceResult:
    """Dub an already-local media file into ``out``; the result as plain data."""
    api_key = state.resolve_api_key()
    transcript = mediafile.resolve_diarized_transcript(
        api_key,
        opts.transcript_id,
        media,
        status_message="Transcribing for dubbing…",
        json_mode=json_mode,
        quiet=state.quiet,
        language_code=opts.source_language,
        # Dub input is typically not English (the API default), so a fresh
        # transcription auto-detects the source language unless --source-lang pins it.
        detect_language=opts.source_language is None,
    )
    transcript_id = str(getattr(transcript, "id", ""))
    utterances = pipeline.utterances_of(transcript, transcript_id)
    translations = pipeline.translate(
        api_key, utterances, language, opts, json_mode=json_mode, quiet=state.quiet
    )
    resolved, speakers = pipeline.assign_voices(utterances, translations, voice_plan, language)
    pipeline.warn_ignored_voice_pins(voice_plan.overrides, speakers, json_mode=json_mode)
    pcm_segments, sample_rate = pipeline.synthesize(
        api_key, resolved, language, json_mode=json_mode, quiet=state.quiet
    )

    # strict=True is an invariant guard only: synthesize returns one PCM per segment.
    placed = [
        (utterance.start_ms, pcm)
        for utterance, pcm in zip(utterances, pcm_segments, strict=True)  # pragma: no mutate
    ]
    track = pipeline.assemble_timeline(placed, sample_rate, pipeline.total_seconds(transcript))
    with tempfile.TemporaryDirectory(prefix="aai-dub-") as tmp:
        wav = Path(tmp) / "dub.wav"
        audio.write_wav(wav, track, sample_rate)
        with output.status("Writing the dubbed file…", json_mode=json_mode, quiet=state.quiet):
            pipeline.mux(ffmpeg, media, wav, out)

    duration = round(pipeline.pcm_seconds(track, sample_rate), 3)
    voices_text = ", ".join(f"{speaker}={voice}" for speaker, voice in speakers.items())
    payload: dict[str, object] = {
        "source": opts.media,
        "out": str(out),
        "language": language,
        "transcript_id": transcript_id,
        "utterances": len(utterances),
        "speakers": speakers,
        "sample_rate": sample_rate,
        "audio_duration_seconds": duration,
    }
    return batch.SourceResult(
        payload=payload,
        summary=f"{out}  dubbed to {language} ({len(utterances)} utterances, {voices_text})",
    )
