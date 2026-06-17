"""Fetch a web page (or PDF) and extract its main readable text.

Backs ``assembly speak --url``: httpx2 (the project's pinned client) fetches the
resource, then the body is narrowed to the readable text. For HTML, trafilatura
strips the boilerplate — nav, sidebars, cookie banners, footers, comment threads
— down to the article body; for a PDF (detected by Content-Type or the ``%PDF-``
magic bytes) pypdf pulls the text layer out of every page. Either way text-to-speech
narrates the piece rather than the page chrome. trafilatura (and its lxml backend)
and pypdf are the heavy imports, so both are deferred to call time to stay off the
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
# Every PDF begins with this signature; the robust signal when a server mislabels
# the Content-Type (e.g. application/octet-stream) or the URL has no .pdf suffix.
_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class Article:
    """The readable content extracted from a web page or PDF."""

    text: str
    title: str | None
    url: str


def fetch_article(url: str) -> Article:
    """Fetch ``url`` and return its main readable text with boilerplate removed.

    HTML pages go through trafilatura; PDFs go through pypdf. Raises a
    :class:`UsageError` when ``url`` isn't an http(s) address or the resource
    yields no readable text, and an :class:`APIError` when the fetch itself
    fails (DNS, timeout, non-2xx).
    """
    if not url.startswith(("http://", "https://")):
        raise UsageError(
            f"Not a web page URL: {url}",
            suggestion="Pass an http(s) URL, e.g. assembly speak --url https://example.com/post.",
        )
    response = _fetch(url)
    content_type = response.headers.get("content-type", "").lower()
    data = response.content
    if _is_pdf(data, content_type):
        text, title = _extract_pdf(data)
        empty_hint = (
            "The PDF may be scanned or image-only — there's no text layer to read "
            "(that needs OCR, which speak doesn't do)."
        )
    else:
        text, title = _extract(response.text)
        empty_hint = "The page may be paywalled, JavaScript-rendered, or not an article."
    if not text:
        raise UsageError(f"Couldn't find readable text at {url}.", suggestion=empty_hint)
    return Article(text=text, title=title, url=url)


def _fetch(url: str) -> httpx.Response:
    """GET ``url``, mapping any network/HTTP failure to APIError.

    Returns the fully-read response so the caller can read it as text (HTML) or
    bytes (PDF) depending on the content type.
    """
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response
    except httpx.HTTPError as exc:
        raise APIError(f"Couldn't fetch {url}: {exc}") from exc


def _is_pdf(data: bytes, content_type: str) -> bool:
    """True when ``data`` is a PDF, by Content-Type or the ``%PDF-`` magic bytes."""
    return "application/pdf" in content_type or data.startswith(_PDF_MAGIC)


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


def _extract_pdf(data: bytes) -> tuple[str | None, str | None]:
    """Pull the text layer and title out of a PDF (pypdf, imported lazily)."""
    from io import BytesIO

    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() for page in reader.pages]
    except PyPdfError as exc:
        raise UsageError(
            f"Couldn't read the PDF at the URL: {exc}",
            suggestion="The file may be encrypted, corrupt, or not a valid PDF.",
        ) from exc
    text = "\n\n".join(page for page in pages if page).strip() or None
    title = getattr(reader.metadata, "title", None)
    return text, title
