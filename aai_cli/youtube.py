from __future__ import annotations

import re
from pathlib import Path

from aai_cli.errors import CLIError

# youtube.com/watch, youtu.be/<id>, music.youtube.com, shorts, with or without scheme.
_YOUTUBE_RE = re.compile(
    r"^(https?://)?(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)",
    re.IGNORECASE,
)


def is_youtube_url(source: str | None) -> bool:
    """True if `source` looks like a YouTube watch/share URL."""
    if not source:
        return False
    return bool(_YOUTUBE_RE.match(source.strip()))


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
