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

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.markup import escape

from aai_cli import mediafile, output, youtube
from aai_cli.commands.dub import _pipeline as pipeline
from aai_cli.context import AppState
from aai_cli.errors import UsageError
from aai_cli.tts import audio, dialogue, session

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
    """Execute one `assembly dub` invocation from already-parsed flags."""
    language = resolve_language(opts.language)
    session.require_available("dub")
    # Parse --voice now: a malformed mapping must fail before the billed pipeline.
    voice_plan = pipeline.VoicePlan(*dialogue.parse_voice_overrides(opts.voice))
    youtube.validate_video_flag(opts.media, video=opts.video)
    youtube.validate_sections_flag(opts.media, opts.download_sections)
    if youtube.is_downloadable_url(opts.media):
        # A media-page URL (YouTube, podcast page, …) is downloaded once — the
        # audio track by default, the full video with --video so the dub keeps
        # the picture, only the --download-sections slices when given — and
        # dubbed locally. ffmpeg is checked before the download so a missing
        # dependency fails before any fetch.
        ffmpeg = mediafile.require_ffmpeg("write the dubbed file")
        downloading = "Downloading video…" if opts.video else "Downloading audio…"
        with tempfile.TemporaryDirectory(prefix="aai-dub-src-") as td:
            with output.status(downloading, json_mode=json_mode, quiet=state.quiet):
                local = youtube.download_media(
                    opts.media,
                    Path(td),
                    video=opts.video,
                    download_sections=opts.download_sections,
                )
            # The download dir is temporary, so the default output lands in the
            # current directory — never next to the temp file.
            out = (
                opts.out
                if opts.out is not None
                else Path.cwd() / default_out_path(local, language).name
            )
            mediafile.validate_out(out, local)
            _dub_and_emit(
                opts, local, out, language, ffmpeg, voice_plan, state, json_mode=json_mode
            )
        return
    if opts.media.startswith(("http://", "https://")):
        raise UsageError(
            "assembly dub can't fetch this URL; it dubs a local file or a "
            "media-page URL yt-dlp can download (YouTube, podcasts, …).",
            suggestion="Download the media first, then dub the local copy.",
        )
    if "://" in opts.media:
        # Path() would collapse the "//" and report a corrupted echo of the URL.
        raise UsageError(
            f"assembly dub needs a local file, not a URL: {opts.media}",
            suggestion="Download the media first, then dub the local copy.",
        )
    media = Path(opts.media)
    mediafile.validate_local_media(media, "dub")
    out = opts.out if opts.out is not None else default_out_path(media, language)
    mediafile.validate_out(out, media)
    ffmpeg = mediafile.require_ffmpeg("write the dubbed file")
    _dub_and_emit(opts, media, out, language, ffmpeg, voice_plan, state, json_mode=json_mode)


def _dub_and_emit(
    opts: DubOptions,
    media: Path,
    out: Path,
    language: str,
    ffmpeg: str,
    voice_plan: pipeline.VoicePlan,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Dub an already-local media file into ``out`` and report the result."""
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
    output.emit(
        payload,
        # language and voices carry user-typed text, so they need escaping too.
        lambda _: output.success(
            f"{escape(str(out))}  dubbed to {escape(language)} "
            f"({len(utterances)} utterances, {escape(voices_text)})"
        ),
        json_mode=json_mode,
    )
