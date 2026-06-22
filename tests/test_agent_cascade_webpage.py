"""Tests for the keyless read-a-URL tool behind `assembly live`.

The tool's only network seam is the injected ``read`` callable, so the whole
fetch -> format flow runs with no sockets (pytest-socket stays armed).
"""

from __future__ import annotations

from aai_cli.agent_cascade import webpage_tool
from aai_cli.core.errors import APIError, UsageError
from aai_cli.core.webpage import Article


def _article(text: str = "Body text.", title: str | None = "Title") -> Article:
    return Article(text=text, title=title, url="https://example.com/post")


# --- _format -----------------------------------------------------------------


def test_format_leads_with_title_then_body():
    out = webpage_tool._format(_article(text="Hello world.", title="My Post"))
    assert out == "My Post\n\nHello world."


def test_format_without_title_is_body_only():
    out = webpage_tool._format(_article(text="Just the body.", title=None))
    assert out == "Just the body."


def test_format_truncates_long_body_with_marker():
    long = "x" * (webpage_tool._MAX_CHARS + 50)
    out = webpage_tool._format(_article(text=long, title=None))
    assert out == "x" * webpage_tool._MAX_CHARS + "\n…[truncated]"


def test_format_keeps_short_body_untruncated():
    out = webpage_tool._format(_article(text="short", title=None))
    assert "[truncated]" not in out
    assert out == "short"


# --- _read (default seam delegates to core.webpage.fetch_article) ------------


def test_read_delegates_to_fetch_article(monkeypatch):
    captured = {}

    def fake_fetch_article(url: str) -> Article:
        captured["url"] = url
        return _article()

    monkeypatch.setattr("aai_cli.core.webpage.fetch_article", fake_fetch_article)
    result = webpage_tool._read("https://example.com/post")
    assert captured["url"] == "https://example.com/post"
    assert result.title == "Title"


# --- build_read_url_tool -----------------------------------------------------


def test_tool_is_named_read_url():
    tool = webpage_tool.build_read_url_tool(read=lambda url: _article())
    assert tool.name == webpage_tool.READ_URL_TOOL_NAME


def test_read_url_happy_path_returns_formatted_text():
    tool = webpage_tool.build_read_url_tool(read=lambda url: _article(text="Article.", title="T"))
    assert tool.invoke({"url": "https://example.com"}) == "T\n\nArticle."


def test_read_url_usage_error_returns_no_readable_text_message():
    def read(url: str) -> Article:
        raise UsageError("Couldn't find readable text.")

    tool = webpage_tool.build_read_url_tool(read=read)
    assert tool.invoke({"url": "https://example.com"}) == (
        "I couldn't find readable text on that page."
    )


def test_read_url_fetch_failure_returns_could_not_read_message():
    def read(url: str) -> Article:
        raise APIError("DNS boom")

    tool = webpage_tool.build_read_url_tool(read=read)
    assert tool.invoke({"url": "https://example.com"}) == ("I couldn't read that page right now.")
