"""Fetch a web page and extract its main article text.

Backs ``assembly speak --url``: httpx2 (the project's pinned client) fetches the
HTML and trafilatura strips the boilerplate — nav, sidebars, cookie banners,
footers, comment threads — down to the readable article body, so text-to-speech
narrates the piece rather than the page chrome. trafilatura (and its lxml
backend) is the heavy import, so it is deferred to call time to stay off the
CLI's startup path.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2 as httpx

from aai_cli.core.errors import APIError, UsageError

# A page fetch shouldn't hang a TTS run; cap it.
_TIMEOUT = 30.0  # pragma: no mutate -- request timeout; nothing observable to assert
# Browser-like UA: some sites serve a stub or block page to unknown clients.
_USER_AGENT = "Mozilla/5.0 (compatible; assembly-cli; +https://www.assemblyai.com)"


@dataclass(frozen=True)
class Article:
    """The readable content extracted from a web page."""

    text: str
    title: str | None
    url: str


def fetch_article(url: str) -> Article:
    """Fetch ``url`` and return its main article text with boilerplate removed.

    Raises a :class:`UsageError` when ``url`` isn't an http(s) address or the
    page yields no readable text, and an :class:`APIError` when the fetch itself
    fails (DNS, timeout, non-2xx).
    """
    if not url.startswith(("http://", "https://")):
        raise UsageError(
            f"Not a web page URL: {url}",
            suggestion="Pass an http(s) URL, e.g. assembly speak --url https://example.com/post.",
        )
    text, title = _extract(_fetch_html(url))
    if not text:
        raise UsageError(
            f"Couldn't find readable text at {url}.",
            suggestion="The page may be paywalled, JavaScript-rendered, or not an article.",
        )
    return Article(text=text, title=title, url=url)


def _fetch_html(url: str) -> str:
    """GET the raw HTML for ``url``, mapping any network/HTTP failure to APIError."""
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as exc:
        raise APIError(f"Couldn't fetch {url}: {exc}") from exc


def _extract(html: str) -> tuple[str | None, str | None]:
    """Pull the main text and title out of ``html`` (trafilatura, imported lazily)."""
    import trafilatura

    text = trafilatura.extract(
        html,
        output_format="txt",
        # Don't narrate the comment thread. trafilatura's comment classifier keys
        # off real-world markup (Disqus, microformats), which synthetic test
        # fixtures can't reproduce, so this flag stays unasserted by the suite.
        include_comments=False,  # pragma: no mutate
    )
    title = getattr(trafilatura.extract_metadata(html), "title", None)
    return text, title
