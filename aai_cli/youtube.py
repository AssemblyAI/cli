"""Downloading audio from media-page URLs (YouTube, podcast pages, …) via yt-dlp.

The AssemblyAI API fetches direct audio URLs itself; this module handles the URLs it
can't — HTML pages whose audio yt-dlp knows how to extract.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from aai_cli.errors import CLIError

# youtube.com/watch, youtu.be/<id>, music.youtube.com, shorts, with or without scheme.
_YOUTUBE_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)",
    re.IGNORECASE,
)

# yt-dlp's default logger prints its own "ERROR: …" line straight to stderr before the
# CLI can raise its one clean error, duplicating the message. Route yt-dlp's output to
# a swallow-everything logger (NullHandler, no propagation) instead.
_YTDLP_LOGGER = logging.getLogger("aai_cli.youtube.yt_dlp")
_YTDLP_LOGGER.addHandler(logging.NullHandler())
_YTDLP_LOGGER.propagate = False


def is_youtube_url(source: str | None) -> bool:
    """True if `source` looks like a YouTube watch/share URL."""
    if not source:
        return False
    return bool(_YOUTUBE_RE.match(source.strip()))


def is_downloadable_url(source: str | None) -> bool:
    """True if `source` is a media-page URL whose audio must be downloaded first.

    YouTube is matched by shape alone — no yt-dlp import needed, so a missing yt-dlp
    still routes to ``download_audio``'s install hint. Other http(s) URLs match when
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


def download_audio(url: str, dest_dir: Path) -> Path:
    """Download the best audio track of `url` into `dest_dir` and return its path.

    Uses yt-dlp; the resulting container (m4a/webm/…) is decodable by ffmpeg
    (streaming) and uploadable for transcription.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise CLIError(
            "YouTube support needs yt-dlp.",
            error_type="ytdlp_missing",
            exit_code=2,
            suggestion="Install it: pip install yt-dlp",
        ) from exc

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _YTDLP_LOGGER,
    }
    try:
        # yt-dlp types `params` as a private `_Params` TypedDict, but a plain options
        # dict is the documented public API; pyright can't reconcile the two.
        with yt_dlp.YoutubeDL(options) as ydl:  # pyright: ignore[reportArgumentType]
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as exc:  # yt-dlp raises many types; surface one clean CLI error
        raise CLIError(
            f"Could not download audio from {url}: {exc}",
            error_type="youtube_error",
            exit_code=1,
        ) from exc

    if not path.is_file():
        # Post-processing can change the extension; fall back to whatever landed.
        # yt-dlp may also drop sidecars (thumbnail, .info.json, a leftover .part)
        # in dest_dir, and iterdir() order is arbitrary, so pick the largest file:
        # the decoded audio track dwarfs any metadata sidecar.
        files = [p for p in dest_dir.iterdir() if p.is_file()]
        if not files:
            raise CLIError(
                f"yt-dlp produced no audio file for {url}.",
                error_type="youtube_error",
                exit_code=1,
            )
        path = max(files, key=lambda p: p.stat().st_size)
    return path
