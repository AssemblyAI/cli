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

import dataclasses
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import assemblyai as aai
from rich.markup import escape

from aai_cli.app import batch, mediafile
from aai_cli.app.context import AppState
from aai_cli.core import client
from aai_cli.core.errors import CLIError, UsageError, mutually_exclusive
from aai_cli.ui import output


@dataclass(frozen=True)
class CaptionOptions:
    """Every `assembly caption` flag as plain data (``--json`` excluded:
    run_command resolves it into the ``json_mode`` argument)."""

    # The raw source as typed: a local path, or a downloadable media-page URL
    # (a pathlib.Path would collapse the "//" in "https://"). Empty in batch mode,
    # where the sources arrive on stdin (--from-stdin) instead of as the argument.
    media: str
    transcript_id: str | None
    chars_per_caption: int | None
    font_size: int | None
    out: Path | None
    from_stdin: bool
    concurrency: int
    force: bool


def default_out_path(media: Path) -> Path:
    """The default output file: ``<stem>.captioned<ext>`` next to the input."""
    return media.parent / f"{media.stem}.captioned{media.suffix}"


# ffmpeg's filtergraph syntax gives these characters meaning (option/filter/chain
# separators, stream labels, quoting), so a path embedded in `-vf subtitles=…`
# must escape them or a TMPDIR containing one would corrupt the filter spec.
_FILTER_ESCAPES = str.maketrans({ch: f"\\{ch}" for ch in "\\':,;[]"})


def subtitles_filter(srt: Path, font_size: int | None) -> str:
    """The ``-vf`` filtergraph burning ``srt`` into the video."""
    # ffmpeg's filtergraph parser takes forward slashes on every platform; a Windows
    # backslash path would otherwise need each separator escaped (and the drive colon
    # mishandled). Normalize to "/" first, then escape the remaining metacharacters
    # (notably the drive ":") — e.g. C:\a\b.srt -> C\:/a/b.srt. No-op on POSIX.
    posix = str(srt).replace(os.sep, "/")
    spec = f"subtitles={posix.translate(_FILTER_ESCAPES)}"
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
        transcript_id = mediafile.transcript_id(transcript)
        raise CLIError(
            f"Transcript {transcript_id} has no captions to burn in.",
            error_type="no_captions",
            exit_code=2,
            suggestion="The media may contain no speech; check it with "
            "'assembly transcribe <file>'.",
        )
    return srt


def run_caption(opts: CaptionOptions, state: AppState, *, json_mode: bool) -> None:
    """Execute `assembly caption`: one source, or a stdin batch (`--from-stdin`)."""
    sources = batch.stdin_sources(opts.media, from_stdin=opts.from_stdin)
    if sources is not None:
        _reject_batch_conflicts(opts)
        batch.run_batch(
            sources,
            worker=_caption_worker(opts, state, force=opts.force, json_mode=json_mode),
            concurrency=opts.concurrency,
            summary_verb="Captioned",
            json_mode=json_mode,
            quiet=state.quiet,
        )
        return
    if not opts.media:
        raise UsageError(
            "Pass a video file or URL to caption, or --from-stdin to read a list from stdin.",
            suggestion="e.g. assembly caption talk.mp4",
        )
    result = _caption_one(opts, state, json_mode=json_mode)
    output.emit(
        result.payload, lambda _: output.success(escape(result.summary)), json_mode=json_mode
    )


def _reject_batch_conflicts(opts: CaptionOptions) -> None:
    """Single-result flags that can't span a many-source batch."""
    mutually_exclusive(
        ("--out", opts.out),
        ("--from-stdin", True),
        suggestion="In batch mode each source gets its own <name>.captioned<ext>; drop --out.",
    )
    mutually_exclusive(
        ("--transcript-id/-t", opts.transcript_id),
        ("--from-stdin", True),
        suggestion="A transcript id can't apply to many sources; drop -t in batch mode.",
    )


def _caption_worker(
    opts: CaptionOptions, state: AppState, *, force: bool, json_mode: bool
) -> batch.Worker:
    """A per-source worker for the batch runner: skip a source whose default output
    already exists (unless ``--force``), else caption it with spinners silenced."""
    quiet_state = dataclasses.replace(state, quiet=True)

    def worker(source: str) -> batch.SourceResult:
        if (
            not force
            and (existing := mediafile.existing_output(source, default_out_path)) is not None
        ):
            return batch.SourceResult(
                payload={"source": source, "out": str(existing)},
                summary=f"{existing} exists",
                status="skipped",
            )
        return _caption_one(
            dataclasses.replace(opts, media=source), quiet_state, json_mode=json_mode
        )

    return worker


def _caption_one(opts: CaptionOptions, state: AppState, *, json_mode: bool) -> batch.SourceResult:
    """Resolve ``opts.media`` to a local video, caption it, and return the result.

    A media-page URL is downloaded once — always the full video, since the
    captions are burned into it.
    """
    ffmpeg = mediafile.require_ffmpeg("burn captions into video")
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
        return _caption_build(opts, media, out, ffmpeg, state, json_mode=json_mode)


def _caption_build(
    opts: CaptionOptions,
    media: Path,
    out: Path,
    ffmpeg: str,
    state: AppState,
    *,
    json_mode: bool,
) -> batch.SourceResult:
    """Caption an already-local video file into ``out``; the result as plain data."""
    transcript = mediafile.resolve_transcript(
        state.resolve_api_key(),
        opts.transcript_id,
        media,
        status_message="Transcribing for captions…",
        json_mode=json_mode,
        quiet=state.quiet,
        config=aai.TranscriptionConfig(),
    )
    transcript_id = mediafile.transcript_id(transcript)
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
    return batch.SourceResult(payload=payload, summary=f"{out}  {captions} caption(s) burned in")
