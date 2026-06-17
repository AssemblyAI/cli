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


def test_fetch_html_returns_body_and_sends_browser_user_agent(monkeypatch):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["ua"] = request.headers["user-agent"]
        return httpx.Response(200, text="<html>ok</html>")

    _client_returning(monkeypatch, handler)
    assert webpage._fetch_html("https://example.com/post") == "<html>ok</html>"
    # The browser-like UA is sent so sites don't serve a stub/block page.
    assert "assembly-cli" in seen["ua"]


def test_fetch_html_follows_redirects(monkeypatch):
    # A 301 must be followed to the final 200; without follow_redirects the
    # client would return the empty 301 body instead of the article.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(301, headers={"Location": "https://example.com/final"})
        return httpx.Response(200, text="final body")

    _client_returning(monkeypatch, handler)
    assert webpage._fetch_html("https://example.com/start") == "final body"


def test_fetch_html_non_2xx_becomes_api_error(monkeypatch):
    _client_returning(monkeypatch, lambda request: httpx.Response(404, text="nope"))
    with pytest.raises(APIError) as exc:
        webpage._fetch_html("https://example.com/missing")
    assert "https://example.com/missing" in exc.value.message


def test_fetch_html_connect_error_becomes_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    _client_returning(monkeypatch, handler)
    with pytest.raises(APIError):
        webpage._fetch_html("https://example.com/post")


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
    monkeypatch.setattr(webpage, "_fetch_html", lambda url: ARTICLE_HTML)
    article = webpage.fetch_article("https://example.com/post")
    assert "first real paragraph of the article body" in article.text
    assert article.title == "The Real Headline"
    assert article.url == "https://example.com/post"


def test_fetch_article_without_readable_text_is_a_usage_error(monkeypatch):
    # A page trafilatura can't extract an article from yields no text -> usage error.
    monkeypatch.setattr(webpage, "_fetch_html", lambda url: "<html><body></body></html>")
    with pytest.raises(UsageError) as exc:
        webpage.fetch_article("https://example.com/empty")
    assert "Couldn't find readable text" in exc.value.message
