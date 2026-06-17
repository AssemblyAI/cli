from __future__ import annotations

import dataclasses

import httpx2 as httpx
import pytest

from aai_cli.core import webpage
from aai_cli.core.errors import APIError, UsageError

# An article wrapped in the usual page chrome: nav, a comment thread, and a
# <title>. The extractor should keep the body and drop the rest.
ARTICLE_HTML = """<!DOCTYPE html><html><head><title>The Real Headline</title></head>
<body>
<nav>Home | About | SubscribeNavBoilerplate</nav>
<article>
<h1>The Real Headline</h1>
<p>This is the first real paragraph of the article body that we care about.</p>
<p>Here is a second substantive paragraph with more content to extract.</p>
</article>
<section class="comments"><p>UserCommentText that appears in the discussion thread below.</p></section>
<footer>FooterBoilerplate copyright 2026.</footer>
</body></html>"""


def _client_returning(monkeypatch, handler):
    """Patch webpage.httpx.Client to route requests through a MockTransport handler
    (the test_eval_data_hf.py pattern) so no real socket is opened."""
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(webpage.httpx, "Client", fake_client)


def test_article_is_immutable():
    # frozen=True: a fetched Article can't be mutated out from under a caller.
    article = webpage.Article(text="body", title="T", url="https://example.com/p")
    # A dynamic field name (not a literal) keeps pyright from resolving the
    # assignment to the read-only attribute — see test_command_options_seam.py.
    field_name = dataclasses.fields(article)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(article, field_name, "tampered")


def test_fetch_returns_body_and_sends_browser_user_agent(monkeypatch):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="<html>ok</html>")

    _client_returning(monkeypatch, handler)
    assert webpage._fetch("https://example.com/post").text == "<html>ok</html>"
    # The browser-like UA is sent so sites don't serve a stub/block page.
    assert "assembly-cli" in seen["ua"]


def test_fetch_follows_redirects(monkeypatch):
    # A 301 must be followed to the final 200; without follow_redirects the
    # client would return the empty 301 body instead of the article.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"Location": "https://example.com/final"})
        return httpx.Response(200, text="final body")

    _client_returning(monkeypatch, handler)
    assert webpage._fetch("https://example.com/start").text == "final body"


def test_fetch_non_2xx_becomes_api_error(monkeypatch):
    _client_returning(monkeypatch, lambda request: httpx.Response(404, text="nope"))
    with pytest.raises(APIError) as exc:
        webpage._fetch("https://example.com/missing")
    assert "https://example.com/missing" in exc.value.message


def test_fetch_connect_error_becomes_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _client_returning(monkeypatch, handler)
    with pytest.raises(APIError):
        webpage._fetch("https://example.com/post")


def test_extract_strips_boilerplate_and_comments_and_reads_title():
    text, title = webpage._extract(ARTICLE_HTML)
    assert text is not None
    # The article body survives...
    assert "first real paragraph of the article body" in text
    # ...while the nav and footer chrome are dropped.
    assert "NavBoilerplate" not in text
    assert "FooterBoilerplate" not in text
    # The <title> drives the extracted title.
    assert title == "The Real Headline"


def test_fetch_article_rejects_non_http_url():
    with pytest.raises(UsageError) as exc:
        webpage.fetch_article("ftp://example.com/file")
    assert "Not a web page URL" in exc.value.message
    assert "http" in (exc.value.suggestion or "")


def test_fetch_article_returns_extracted_text_and_title(monkeypatch):
    _client_returning(
        monkeypatch,
        lambda request: httpx.Response(
            200, text=ARTICLE_HTML, headers={"content-type": "text/html; charset=utf-8"}
        ),
    )
    article = webpage.fetch_article("https://example.com/post")
    assert "first real paragraph of the article body" in article.text
    assert article.title == "The Real Headline"
    assert article.url == "https://example.com/post"


def test_fetch_article_without_readable_text_is_a_usage_error(monkeypatch):
    # A page trafilatura can't extract an article from yields no text -> usage error.
    _client_returning(
        monkeypatch,
        lambda request: httpx.Response(
            200, text="<html><body></body></html>", headers={"content-type": "text/html"}
        ),
    )
    with pytest.raises(UsageError) as exc:
        webpage.fetch_article("https://example.com/empty")
    assert "Couldn't find readable text" in exc.value.message
    # The HTML-specific hint, not the scanned-PDF one.
    assert "paywalled" in (exc.value.suggestion or "")


def _make_pdf(body_text: str, *, title: str | None = None) -> bytes:
    """Build a minimal one-page PDF whose content stream shows ``body_text``.

    Offsets in the xref table are computed as the file is assembled so pypdf reads
    it without falling back to recovery — enough of a real PDF to exercise the
    text-layer extraction path end to end.
    """
    content = b"BT /F1 24 Tf 72 120 Td (" + body_text.encode("latin-1") + b") Tj ET"
    info = b"<< /Title (" + title.encode("latin-1") + b") >>" if title else None
    page = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page,
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    if info is not None:
        objects.append(info)
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    trailer = b"<< /Size %d /Root 1 0 R" % (len(objects) + 1)
    if info is not None:
        trailer += b" /Info %d 0 R" % len(objects)
    trailer += b" >>"
    out += b"trailer\n%s\nstartxref\n%d\n%%%%EOF" % (trailer, xref_pos)
    return bytes(out)


def _pdf_response(data: bytes, content_type: str = "application/pdf") -> httpx.Response:
    return httpx.Response(200, content=data, headers={"content-type": content_type})


def test_is_pdf_detects_by_content_type_and_magic_bytes():
    # Either signal alone is sufficient...
    assert webpage._is_pdf(b"not a pdf", "application/pdf; charset=binary")
    assert webpage._is_pdf(b"%PDF-1.7\n...", "application/octet-stream")
    # ...and an HTML response is not a PDF.
    assert not webpage._is_pdf(b"<html></html>", "text/html; charset=utf-8")


def test_fetch_article_extracts_pdf_text_and_title(monkeypatch):
    pdf = _make_pdf("Hello from the PDF body text", title="A PDF Report")
    _client_returning(monkeypatch, lambda request: _pdf_response(pdf))
    article = webpage.fetch_article("https://example.com/report")
    assert "Hello from the PDF body text" in article.text
    assert article.title == "A PDF Report"
    assert article.url == "https://example.com/report"


def test_fetch_article_dispatches_pdf_by_magic_bytes_despite_generic_type(monkeypatch):
    # A server mislabeling the PDF as octet-stream still takes the PDF path.
    pdf = _make_pdf("Magic-byte routed body")
    _client_returning(
        monkeypatch, lambda request: _pdf_response(pdf, content_type="application/octet-stream")
    )
    assert "Magic-byte routed body" in webpage.fetch_article("https://example.com/x").text


def test_fetch_article_scanned_pdf_without_text_is_a_usage_error(monkeypatch):
    # An image-only PDF has no text layer -> usage error with the OCR-shaped hint.
    pdf = _make_pdf("")
    _client_returning(monkeypatch, lambda request: _pdf_response(pdf))
    with pytest.raises(UsageError) as exc:
        webpage.fetch_article("https://example.com/scanned.pdf")
    assert "Couldn't find readable text" in exc.value.message
    assert "scanned" in (exc.value.suggestion or "")


def test_fetch_article_corrupt_pdf_is_a_usage_error(monkeypatch):
    # Passes the %PDF- magic check but isn't a parseable PDF -> usage error.
    _client_returning(monkeypatch, lambda request: _pdf_response(b"%PDF-1.4\nnot really a pdf"))
    with pytest.raises(UsageError) as exc:
        webpage.fetch_article("https://example.com/broken.pdf")
    assert "PDF" in exc.value.message
