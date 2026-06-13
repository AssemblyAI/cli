"""Run logic for `assembly caption`: transcribe → SRT export → ffmpeg burn-in.

The command module (aai_cli/commands/caption/__init__.py) only parses argv — it builds a
``CaptionOptions`` and hands it to ``run_caption`` via ``context.run_command``
(the options/run split, see AGENTS.md), so tests drive the whole pipeline by
constructing options directly.

The pipeline: the video is transcribed (or an existing transcript is reused via
``--transcript-id``), the transcript's SRT captions are fetched from the export
endpoint, and ffmpeg's ``subtitles`` filter burns them into the picture (open
captions, always visible) while the audio stream is copied untouched. A
YouTube/media-page URL is downloaded first — always the full video, since the
captions are burned into it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import assemblyai as aai
from rich.markup import escape

from aai_cli import client, mediafile, output
from aai_cli.context import AppState
from aai_cli.errors import CLIError


@dataclass(frozen=True)
class CaptionOptions:
    """Every `assembly caption` flag as plain data (``--json`` excluded:
    run_command resolves it into the ``json_mode`` argument)."""

    # The raw source as typed: a local path, or a downloadable media-page URL
    # (a pathlib.Path would collapse the "//" in "https://").
    media: str
    transcript_id: str | None
    chars_per_caption: int | None
    font_size: int | None
    out: Path | None


def default_out_path(media: Path) -> Path:
    """The default output file: ``<stem>.captioned<ext>`` next to the input."""
    return media.parent / f"{media.stem}.captioned{media.suffix}"


# ffmpeg's filtergraph syntax gives these characters meaning (option/filter/chain
# separators, stream labels, quoting), so a path embedded in `-vf subtitles=…`
# must escape them or a TMPDIR containing one would corrupt the filter spec.
_FILTER_ESCAPES = str.maketrans({ch: f"\\{ch}" for ch in "\\':,;[]"})


def subtitles_filter(srt: Path, font_size: int | None) -> str:
    """The ``-vf`` filtergraph burning ``srt`` into the video."""
    spec = f"subtitles={str(srt).translate(_FILTER_ESCAPES)}"
    if font_size is not None:
        spec += f":force_style=FontSize={font_size}"
    return spec


def _burn(ffmpeg: str, media: Path, srt: Path, out: Path, font_size: int | None) -> None:
    """Burn the ``srt`` captions into ``media``'s video stream, writing ``out``.

    The video is necessarily re-encoded (the captions become pixels); ``-c:a
    copy`` carries the audio over untouched. The explicit ``-map 0:v`` makes
    audio-only input an ffmpeg error ("matches no streams") instead of a silent
    uncaptioned copy; ``-map 0:a?`` keeps a silent video legal. ``-y`` makes a
    re-run overwrite its own earlier output instead of stalling on ffmpeg's
    prompt.
    """
    result = mediafile.run_ffmpeg(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(media),
            "-vf",
            subtitles_filter(srt, font_size),
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:a",
            "copy",
            mediafile.path_arg(out),
        ]
    )
    if result.returncode != 0:
        raise mediafile.ffmpeg_failure(
            result,
            "write",
            out,
            error_type="caption_failed",
            suggestion="Check that the input is a readable video file — captions "
            "can't be burned into audio-only media.",
        )


def _fetch_srt(transcript: object, opts: CaptionOptions, *, json_mode: bool, quiet: bool) -> str:
    """The transcript's SRT captions from the export endpoint; empty is an error."""
    with output.status("Fetching captions…", json_mode=json_mode, quiet=quiet):
        srt = client.select_transcript_field(
            transcript, "srt", chars_per_caption=opts.chars_per_caption
        )
    if not srt.strip():
        transcript_id = str(getattr(transcript, "id", ""))
        raise CLIError(
            f"Transcript {transcript_id} has no captions to burn in.",
            error_type="no_captions",
            exit_code=2,
            suggestion="The media may contain no speech; check it with "
            "'assembly transcribe <file>'.",
        )
    return srt


def run_caption(opts: CaptionOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute one `assembly caption` invocation from already-parsed flags."""
    ffmpeg = mediafile.require_ffmpeg("burn captions into video")
    # A media-page URL is downloaded once — always the full video, since the
    # captions are burned into it.
    with mediafile.resolve_media_source(
        opts.media,
        "caption",
        fetch_clause="captions a local file or a media-page URL yt-dlp can download (YouTube, …)",
        download_suggestion="Download the video first, then caption the local copy.",
        video=True,
        download_sections=None,
        json_mode=json_mode,
        quiet=state.quiet,
    ) as (media, downloaded):
        if not downloaded:
            mediafile.validate_local_media(media, "caption", kind="video")
        out = mediafile.default_output(
            opts.out, media, downloaded=downloaded, namer=default_out_path
        )
        mediafile.validate_out(out, media)
        _caption_and_emit(opts, media, out, ffmpeg, state, json_mode=json_mode)


def _caption_and_emit(
    opts: CaptionOptions,
    media: Path,
    out: Path,
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> None:
    """Caption an already-local video file into ``out`` and report the result."""
    transcript = mediafile.resolve_transcript(
        state.resolve_api_key(),
        opts.transcript_id,
        media,
        status_message="Transcribing for captions…",
        json_mode=json_mode,
        quiet=state.quiet,
        config=aai.TranscriptionConfig(),
    )
    transcript_id = str(getattr(transcript, "id", ""))
    srt = _fetch_srt(transcript, opts, json_mode=json_mode, quiet=state.quiet)
    captions = srt.count("-->")  # one arrow per SRT cue timing line
    with tempfile.TemporaryDirectory(prefix="aai-caption-") as tmp:
        srt_path = Path(tmp) / "captions.srt"
        srt_path.write_text(srt, encoding="utf-8")
        with output.status("Burning captions…", json_mode=json_mode, quiet=state.quiet):
            _burn(ffmpeg, media, srt_path, out, opts.font_size)
    payload: dict[str, object] = {
        "source": opts.media,
        "out": str(out),
        "transcript_id": transcript_id,
        "captions": captions,
    }
    output.emit(
        payload,
        lambda _: output.success(f"{escape(str(out))}  {captions} caption(s) burned in"),
        json_mode=json_mode,
    )
