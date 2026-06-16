"""Podcast RSS/Atom feed expansion for ``assembly transcribe``.

A feed URL names a whole show, so transcribing it means transcribing every
episode. ``feed_episode_urls`` fetches the URL and, when ``feedparser`` recognizes
it as an RSS or Atom feed, returns its episode enclosure URLs (in feed order —
newest first) for the batch path to transcribe, one resumable sidecar per episode.
The enclosures are direct media URLs the API fetches itself, so — unlike a YouTube
or podcast *page*, which yt-dlp downloads first — no local download step is needed.

Detection is deliberately narrow so a direct media URL or ordinary web page still
falls through to the single-source path untouched (and is never fetched twice):
only an http(s) URL whose path is feed-shaped — no extension, or one of
``.xml``/``.rss``/``.atom`` — and that no dedicated yt-dlp extractor already claims
is sniffed, the response body is bounded, and only content ``feedparser`` parses as
a real feed with at least one enclosure is treated as a feed. We hand ``feedparser``
the already-fetched bytes (never the URL) so our bounded, safe fetch below stays the
only network path.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlsplit

from pydantic import BaseModel

from aai_cli.core import youtube

# A feed lives at an extensionless URL (e.g. feeds.simplecast.com/<id>) or a feed
# document (.xml/.rss/.atom). Every other path — .mp3, .txt, .pdf — is never a feed,
# so it is left for the single-source path and never fetched here.
_FEED_URL_SUFFIXES = frozenset({"", ".xml", ".rss", ".atom"})

# Bound the download so a hostile or huge URL can't exhaust memory; 10 MB of feed
# already holds thousands of episodes, far past any realistic batch.
_MAX_FEED_BYTES = 10 * 1024 * 1024  # pragma: no mutate -- tuning knob, not behavior
_FETCH_TIMEOUT_SECONDS = 15.0  # pragma: no mutate -- tuning knob, not behavior


class _Enclosure(BaseModel):
    """One ``<enclosure>`` / Atom enclosure link; ``href`` is the media URL."""

    href: str = ""


class _Entry(BaseModel):
    enclosures: list[_Enclosure] = []


class _ParsedFeed(BaseModel):
    """The slice of feedparser's untyped result we use, validated into a real type
    (the project pattern for untyped third-party returns — cf. core/wer.py)."""

    # feedparser sets ``version`` to a non-empty id ("rss20", "atom10", …) for a
    # recognized feed and to "" for anything it doesn't recognize as one.
    version: str = ""
    entries: list[_Entry] = []


def feed_episode_urls(url: str) -> list[str] | None:
    """The episode media URLs if `url` is a podcast feed, else ``None``.

    Returns ``None`` (stay single-source) for a direct-media URL, a yt-dlp page,
    an unreachable URL, or any content that isn't a feed carrying enclosures.
    """
    if not _looks_like_feed_url(url) or youtube.is_downloadable_url(url):
        return None
    body = _fetch(url)
    if body is None:
        return None
    return _episode_urls(body)


def _looks_like_feed_url(url: str) -> bool:
    """True when the URL path is feed-shaped: extensionless or a feed document."""
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    return suffix in _FEED_URL_SUFFIXES


def _episode_urls(body: str) -> list[str] | None:
    """The enclosure URLs in a feed body, deduped in document order; ``None`` when
    feedparser doesn't recognize it as a feed or it carries no enclosures."""
    import feedparser

    # feedparser ships only partial inline types (its parse signature is Unknown),
    # so the result is validated through _ParsedFeed below; mirror remotefs.py's
    # fsspec shim in ignoring the unavoidable unknown-member report on the call.
    raw = feedparser.parse(body)  # pyright: ignore[reportUnknownMemberType]
    parsed = _ParsedFeed.model_validate(raw)
    if not parsed.version:
        return None
    urls = [enc.href for entry in parsed.entries for enc in entry.enclosures if enc.href]
    deduped = list(dict.fromkeys(urls))
    return deduped or None


def _fetch(url: str) -> str | None:
    """Up to ``_MAX_FEED_BYTES`` of `url` decoded as text, or ``None`` on any failure
    or when the response is obviously binary media (audio/video/image)."""
    import httpx2 as httpx

    chunks: list[bytes] = []
    try:
        with (
            httpx.Client(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client,
            client.stream("GET", url) as response,
        ):
            if not response.is_success:
                return None
            content_type = response.headers.get("content-type", "").lower()
            if content_type.startswith(("audio/", "video/", "image/")):
                return None
            total = 0
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                total += len(chunk)
                if total >= _MAX_FEED_BYTES:
                    break
    except (httpx.HTTPError, OSError):
        return None
    return b"".join(chunks).decode("utf-8", "replace")
