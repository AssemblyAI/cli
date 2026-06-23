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
from urllib.parse import urljoin

import httpx2 as httpx

from aai_cli.core import ssrf
from aai_cli.core.errors import APIError, UsageError

# A page fetch shouldn't hang a TTS run; cap it.
_TIMEOUT = 30.0  # pragma: no mutate -- request timeout; nothing observable to assert
# Bound the body so a hostile or huge URL can't exhaust memory; 10 MB is far past any
# article or PDF we'd narrate, and the cap is what makes the fetch memory-safe.
_MAX_BYTES = 10 * 1024 * 1024  # pragma: no mutate -- tuning knob, not behavior
# Browser-like UA: some sites serve a stub or block page to unknown clients.
_USER_AGENT = "Mozilla/5.0 (compatible; assembly-cli; +https://www.assemblyai.com)"
# Every PDF begins with this signature; the robust signal when a server mislabels
# the Content-Type (e.g. application/octet-stream) or the URL has no .pdf suffix.
_PDF_MAGIC = b"%PDF-"


@dataclass(frozen=True)
class _Page:
    """A fetched resource: size-capped raw bytes (for the PDF path), its
    content-type, and the bytes decoded with the response's own charset (for HTML)."""

    content: bytes
    content_type: str
    text: str


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
    page = _fetch(url)
    if _is_pdf(page.content, page.content_type):
        text, title = _extract_pdf(page.content)
        empty_hint = (
            "The PDF may be scanned or image-only — there's no text layer to read "
            "(that needs OCR, which speak doesn't do)."
        )
    else:
        text, title = _extract(page.text)
        empty_hint = "The page may be paywalled, JavaScript-rendered, or not an article."
    if not text:
        raise UsageError(f"Couldn't find readable text at {url}.", suggestion=empty_hint)
    return Article(text=text, title=title, url=url)


def _fetch(url: str) -> _Page:
    """GET ``url`` as a size-capped, SSRF-checked :class:`_Page`.

    Redirects are followed manually so the SSRF guard runs on **every hop** (a
    public URL can redirect to an internal one). A network/HTTP failure maps to
    APIError; a non-public host raises :class:`ssrf.BlockedURLError`.
    """
    current = url
    try:
        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            for _ in range(ssrf.MAX_REDIRECTS + 1):  # pragma: no mutate -- hop cap; +-1 equivalent
                ssrf.assert_public_url(current)
                with client.stream("GET", current) as response:
                    if response.status_code in ssrf.REDIRECT_STATUS:
                        location = response.headers.get("location")
                        if location:
                            current = urljoin(current, location)
                            continue
                    response.raise_for_status()
                    return _read_page(response)
            raise APIError(f"Too many redirects fetching {url}.")
    except httpx.HTTPError as exc:
        raise APIError(f"Couldn't fetch {url}: {exc}") from exc
    except OSError as exc:
        raise APIError(f"Couldn't resolve {url}: {exc}") from exc


def _read_page(response: httpx.Response) -> _Page:
    """Read at most ``_MAX_BYTES`` of ``response`` so a huge body can't exhaust memory."""
    chunks: list[bytes] = []
    # The running total + early break bound *memory* (stop reading a huge stream); the
    # output is bounded by the slice below, which is what the test asserts — so the
    # counter itself is not output-observable and can't be killed by a mutation test.
    total = 0  # pragma: no mutate -- memory-bound counter; output is sliced below
    for chunk in response.iter_bytes():
        chunks.append(chunk)
        total += len(chunk)  # pragma: no mutate -- memory-bound counter; output is sliced below
        if total >= _MAX_BYTES:  # pragma: no mutate -- memory bound; can't be asserted by OOM
            break
    content = b"".join(chunks)[:_MAX_BYTES]
    content_type = response.headers.get("content-type", "").lower()
    # Decode with the charset httpx parsed from the Content-Type header (the same
    # source response.text would use), falling back to UTF-8 for an unlabeled page.
    text = content.decode(response.charset_encoding or "utf-8", errors="replace")
    return _Page(content=content, content_type=content_type, text=text)


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
