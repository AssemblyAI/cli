"""Downloading media from media-page URLs (YouTube, podcast pages, …) via yt-dlp.

The AssemblyAI API fetches direct audio URLs itself; this module handles the URLs it
can't — HTML pages whose media yt-dlp knows how to extract. Downloads fetch the
audio track by default (all the API needs); commands whose output is the media
itself (clip/dub/caption) can fetch the full video instead.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from aai_cli.core.errors import CLIError, UsageError

# youtube.com/watch, youtu.be/<id>, music.youtube.com, shorts, with or without scheme.
_YOUTUBE_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)",
    re.IGNORECASE,
)

# yt-dlp's own --download-sections timestamp grammar (verbatim): an optional signed
# start, a "-" separator, and an optional signed end. Both sides may be omitted
# (defaulting to 0 / inf); a leading "-" negates a bound (offset from the end).
_SECTION_RANGE_RE = re.compile(
    r"""(?x)(?:
        (?P<start_sign>-?)(?P<start>[^-]+)
    )?\s*-\s*(?:
        (?P<end_sign>-?)(?P<end>[^-]+)
    )?"""
)

# yt-dlp appends report-a-bug boilerplate ("please report this issue on
# https://github.com/yt-dlp/yt-dlp/issues… Confirm you are on the latest version using
# yt-dlp -U") to most extractor errors; it isn't actionable for CLI users, so it is
# trimmed off before the message reaches our one clean error line.
_YTDLP_BUG_REPORT_RE = re.compile(r";?\s*please report this issue on .*", re.IGNORECASE | re.DOTALL)


def _ytdlp_error_message(exc: BaseException) -> str:
    """The meaningful part of a yt-dlp failure: its message without the trailing
    report-a-bug boilerplate or the redundant ``ERROR:`` prefix."""
    text = _YTDLP_BUG_REPORT_RE.sub("", str(exc)).strip()
    return text.removeprefix("ERROR:").strip() or str(exc).strip()


# yt-dlp's default logger prints its own "ERROR: …" line straight to stderr before the
# CLI can raise its one clean error, duplicating the message. Route yt-dlp's output to
# a swallow-everything logger (NullHandler, no propagation) instead.
_YTDLP_LOGGER = logging.getLogger("aai_cli.core.youtube.yt_dlp")
_YTDLP_LOGGER.addHandler(logging.NullHandler())
_YTDLP_LOGGER.propagate = False


def is_youtube_url(source: str | None) -> bool:
    """True if `source` looks like a YouTube watch/share URL."""
    if not source:
        return False
    return bool(_YOUTUBE_RE.match(source.strip()))


def is_downloadable_url(source: str | None) -> bool:
    """True if `source` is a media-page URL whose media must be downloaded first.

    YouTube is matched by shape alone — no yt-dlp import needed, so a missing yt-dlp
    still routes to ``download_media``'s install hint. Other http(s) URLs match when
    a dedicated yt-dlp extractor claims them (Apple Podcasts, Spreaker, SoundCloud,
    …). Direct audio URLs and unknown pages match only yt-dlp's catch-all ``Generic``
    extractor, which is excluded: those pass through untouched for the API to fetch.
    """
    if is_youtube_url(source):
        return True
    url = (source or "").strip()
    if not url.startswith(("http://", "https://")):
        # Local paths (and other non-URL sources) never need a download; skipping the
        # extractor sweep also avoids importing yt-dlp on the common local-file path.
        return False
    return _ytdlp_extractor_claims(url)


def _ytdlp_extractor_claims(url: str) -> bool:
    """True if a dedicated (non-``Generic``) yt-dlp extractor matches `url`."""
    try:
        from yt_dlp.extractor import gen_extractor_classes
    except ImportError:
        return False
    return any(ie.suitable(url) and ie.ie_key() != "Generic" for ie in gen_extractor_classes())


def _section_timestamp(value: str, sign: str | None) -> float:
    """Seconds for one side of a ``--download-sections`` range; a "-" sign negates it."""
    from yt_dlp.utils import parse_duration

    seconds = float("inf") if value in ("inf", "infinite") else parse_duration(value)
    # parse_duration returns None for an unparseable timestamp, but yt-dlp leaves it
    # untyped and pyright over-narrows the inferred return to ``float``.
    if seconds is None:  # pyright: ignore[reportUnnecessaryComparison]
        raise UsageError(
            f'Invalid --download-sections time "{value}".',
            suggestion='Use the form "*start-end", e.g. "*0:00-5:00".',
        )
    return -seconds if sign else seconds


def _section_range(text: str) -> tuple[float, float]:
    """Parse one ``*``-stripped ``start-end`` range into (start, end) seconds."""
    match = _SECTION_RANGE_RE.fullmatch(text) if text != "-" else None
    if match is None:
        raise UsageError(
            f'Invalid --download-sections time range "{text}".',
            suggestion='Use the form "*start-end", e.g. "*0:00-5:00".',
        )
    start = _section_timestamp(match.group("start") or "0", match.group("start_sign"))
    end = _section_timestamp(match.group("end") or "inf", match.group("end_sign"))
    if end == float("-inf"):
        raise UsageError('"-inf" is not a valid --download-sections end.')
    return start, end


def parse_download_sections(
    sections: list[str],
) -> tuple[list[str], list[tuple[float, float]], bool]:
    """Split yt-dlp ``--download-sections`` specs into (chapter regexes, ranges, from-url).

    Mirrors yt-dlp's own grammar: ``*from-url`` keeps the source's own chapter range, a
    ``*``-prefixed spec is one or more comma-separated ``start-end`` timestamp ranges, and
    anything else is a chapter-title regex. Raises ``UsageError`` on a malformed spec.
    """
    chapters: list[str] = []
    ranges: list[tuple[float, float]] = []
    from_url = False
    for spec in sections:
        if spec == "*from-url":
            from_url = True
        elif spec.startswith("*"):
            ranges.extend(_section_range(chunk.strip()) for chunk in spec[1:].split(","))
        else:
            try:
                re.compile(spec)
            except re.error as err:
                raise UsageError(f'Invalid --download-sections regex "{spec}": {err}.') from err
            chapters.append(spec)
    return chapters, ranges, from_url


def validate_video_flag(source: str, *, video: bool) -> None:
    """Reject ``--video`` for a source that isn't a downloadable URL.

    The flag selects the full-video download for a media-page URL; a local file's
    video stream is already operated on directly, so the flag would be a silent
    no-op there — and a requested flag is never dropped silently.
    """
    if video and not is_downloadable_url(source):
        raise UsageError(
            "--video only applies to a downloadable URL source (YouTube, media pages, …).",
            suggestion="A local file's video is used directly already; drop --video.",
        )


def validate_sections_flag(source: str, sections: list[str]) -> None:
    """Reject ``--download-sections`` for a source that isn't a downloadable URL.

    The specs select which parts of a media-page download yt-dlp fetches; a local
    file (or a direct URL the API fetches itself) is never downloaded, so the flag
    would be a silent no-op there — and a requested flag is never dropped silently.
    """
    if sections and not is_downloadable_url(source):
        raise UsageError(
            "--download-sections only applies to a downloadable URL source "
            "(YouTube, media pages, …).",
            suggestion="Cut a local file first (assembly clip <file> --range START-END), "
            "then use the cut.",
        )


def _ytdlp_options(
    dest_dir: Path, *, video: bool, download_sections: list[str] | None
) -> dict[str, object]:
    """The yt-dlp options for one download (caller has already imported yt_dlp)."""
    options: dict[str, object] = {
        "format": "bestvideo*+bestaudio/best" if video else "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _YTDLP_LOGGER,
    }
    if video:
        # Separate video+audio streams (YouTube's norm) must merge into one
        # container; mp4 keeps the result playable (and clippable) everywhere.
        options["merge_output_format"] = "mp4"
    if download_sections:
        from yt_dlp.utils import download_range_func

        chapters, ranges, from_url = parse_download_sections(download_sections)
        # yt-dlp types ``ranges`` as int tuples, but its own code (and ours) uses float
        # seconds; the dict value is loosely typed for YoutubeDL either way.
        options["download_ranges"] = download_range_func(
            [re.compile(chapter) for chapter in chapters],
            ranges,  # pyright: ignore[reportArgumentType]
            from_url,
        )
        # Cut at exact timestamps rather than the nearest keyframe (yt-dlp's default).
        options["force_keyframes_at_cuts"] = True
    return options


def download_media(
    url: str,
    dest_dir: Path,
    *,
    video: bool = False,
    download_sections: list[str] | None = None,
) -> Path:
    """Download the media of `url` into `dest_dir` and return its path.

    Uses yt-dlp. The default fetches only the best audio track — all transcription
    needs; the resulting container (m4a/webm/…) is decodable by ffmpeg (streaming)
    and uploadable. ``video=True`` fetches the full video instead (best video+audio,
    merged to mp4) for commands whose output is the media itself. `download_sections`
    accepts yt-dlp's ``--download-sections`` specs (e.g. ``*0:00-5:00`` for the first
    five minutes), each fetching only that part of the source.
    """
    noun = "video" if video else "audio"
    try:
        import yt_dlp
    except ImportError as exc:
        raise CLIError(
            "YouTube support needs yt-dlp.",
            error_type="ytdlp_missing",
            exit_code=2,
            suggestion="Install it: pip install yt-dlp",
        ) from exc

    options = _ytdlp_options(dest_dir, video=video, download_sections=download_sections)
    try:
        # yt-dlp types `params` as a private `_Params` TypedDict, but a plain options
        # dict is the documented public API; pyright can't reconcile the two.
        with yt_dlp.YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt-dlp raises many types; surface one clean CLI error
        raise CLIError(
            f"Could not download {noun} from {url}: {_ytdlp_error_message(exc)}",
            error_type="youtube_error",
            exit_code=1,
        ) from exc

    if not path.is_file():
        # Post-processing can change the extension; fall back to whatever landed.
        # yt-dlp may also drop sidecars (thumbnail, .info.json, a leftover .part)
        # in dest_dir, and iterdir() order is arbitrary, so pick the largest file:
        # the decoded media track dwarfs any metadata sidecar.
        files = [p for p in dest_dir.iterdir() if p.is_file()]
        if not files:
            raise CLIError(
                f"yt-dlp produced no {noun} file for {url}.",
                error_type="youtube_error",
                exit_code=1,
            )
        path = max(files, key=lambda p: p.stat().st_size)
    return path
