"""Podcast RSS/Atom feed expansion for ``assembly transcribe``.

A feed URL names a whole show, so transcribing it means transcribing every
episode. ``feed_episode_urls`` fetches the URL and, when it parses as an RSS or
Atom feed carrying audio/video enclosures, returns those enclosure URLs (in feed
order — newest first) for the batch path to transcribe, one resumable sidecar per
episode. The enclosures are direct media URLs the API fetches itself, so — unlike
a YouTube or podcast *page*, which yt-dlp downloads first — no local download step
is needed.

Detection is deliberately narrow so a direct media URL or ordinary web page still
falls through to the single-source path untouched (and is never fetched twice):
only an http(s) URL whose path is feed-shaped — no extension, or one of
``.xml``/``.rss``/``.atom`` — and that no dedicated yt-dlp extractor already claims
is sniffed, the response body is bounded, and only content that actually parses as
a feed with at least one enclosure is treated as a feed.
"""

from __future__ import annotations

import html
import re
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from aai_cli.core import youtube

# A feed lives at an extensionless URL (e.g. feeds.simplecast.com/<id>) or a feed
# document (.xml/.rss/.atom). Every other path — .mp3, .txt, .pdf — is never a feed,
# so it is left for the single-source path and never fetched here.
_FEED_URL_SUFFIXES = frozenset({"", ".xml", ".rss", ".atom"})

# Bound the download so a hostile or huge URL can't exhaust memory; 10 MB of feed
# already holds thousands of episodes, far past any realistic batch.
_MAX_FEED_BYTES = 10 * 1024 * 1024
_FETCH_TIMEOUT_SECONDS = 15.0

# A feed body must announce itself with an <rss …> or <feed …> root element
# (namespaced or not) before its <enclosure>s are trusted, so a stray HTML page
# that merely contains the word "enclosure" is never mistaken for a podcast.
_FEED_ROOT_RE = re.compile(r"<\s*(?:[\w.-]+:)?(?:rss|feed)\b", re.IGNORECASE)
# RSS 2.0 episodes: <enclosure url="…" type="audio/mpeg" length="…"/>. The url
# attribute can sit anywhere in the tag and use either quote style.
_ENCLOSURE_TAG_RE = re.compile(r"<\s*enclosure\b([^>]*)>", re.IGNORECASE)
# Atom episodes: <link rel="enclosure" href="…"/> (rel/href in either order).
_LINK_TAG_RE = re.compile(r"<\s*link\b([^>]*)>", re.IGNORECASE)
_URL_ATTR_RE = re.compile(r"""\burl\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_HREF_ATTR_RE = re.compile(r"""\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_REL_ENCLOSURE_RE = re.compile(r"""\brel\s*=\s*["']enclosure["']""", re.IGNORECASE)


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
    """The enclosure URLs in a feed body, deduped in document order; ``None`` when it
    isn't a feed or carries no enclosures."""
    if not _FEED_ROOT_RE.search(body):
        return None
    urls = [*_rss_enclosure_urls(body), *_atom_enclosure_urls(body)]
    deduped = list(dict.fromkeys(u for u in urls if u))
    return deduped or None


def _rss_enclosure_urls(body: str) -> list[str]:
    """The ``url`` of every RSS ``<enclosure url="…">`` tag, HTML-unescaped."""
    return [
        html.unescape(match.group(1).strip())
        for attrs in _ENCLOSURE_TAG_RE.findall(body)
        if (match := _URL_ATTR_RE.search(attrs)) is not None
    ]


def _atom_enclosure_urls(body: str) -> list[str]:
    """The ``href`` of every Atom ``<link rel="enclosure" href="…">``, HTML-unescaped."""
    return [
        html.unescape(match.group(1).strip())
        for attrs in _LINK_TAG_RE.findall(body)
        if _REL_ENCLOSURE_RE.search(attrs) is not None
        and (match := _HREF_ATTR_RE.search(attrs)) is not None
    ]


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
