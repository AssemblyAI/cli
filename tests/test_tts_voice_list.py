"""The streaming-TTS voice listing (aai_cli/tts/voice_list.py): the built-in voices
plus the account's own from GET /v1/voices — the request shape, pagination, error
normalization, response parsing, and the human rendering."""

from __future__ import annotations

import httpx2 as httpx
import pytest

from aai_cli.core import environments
from aai_cli.core.errors import APIError, NotAuthenticated
from aai_cli.tts import voice_list, voices

# Records in the documented shape (abridged): the slug is the voice_id to send.
OWN = [
    {"id": "01a1", "slug": "alex-6f2f2e", "name": "Alex", "language": "english"},
    {"id": "01a2", "slug": "bea-12ab34", "name": "Bea", "language": "spanish"},
]


@pytest.fixture(autouse=True)
def _sandbox():
    environments.set_active(environments.get("sandbox000"))


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client
    seen_kwargs = {}

    def fake_client(*args, **kwargs):
        seen_kwargs.update(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(voice_list.httpx, "Client", fake_client)
    return seen_kwargs


def _respond(status: int, *, json: object = None, text: str | None = None):
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["auth"] = request.headers.get("authorization")
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=json)

    return handler, seen


def _page(listing, *, cursor=None):
    return {"voices": listing, "has_more": cursor is not None, "next_cursor": cursor}


def test_voices_url_is_the_active_tts_host():
    assert voice_list.voices_url() == (
        "https://streaming-tts.sandbox000.assemblyai-labs.com/v1/voices"
    )


def test_fetch_gets_the_account_voices_with_the_raw_api_key(monkeypatch):
    handler, seen = _respond(200, json=_page(OWN))
    client_kwargs = _patch_transport(monkeypatch, handler)
    assert voice_list.fetch_voices("sk_voices") == OWN
    assert seen["method"] == "GET"
    # The default scope (the account's own voices), at the largest page size.
    assert seen["url"] == (
        "https://streaming-tts.sandbox000.assemblyai-labs.com/v1/voices?limit=100"
    )
    # Streaming TTS takes the raw key, not a Bearer token — same as its socket.
    assert seen["auth"] == "sk_voices"
    assert client_kwargs["timeout"] == 15.0


def test_fetch_follows_next_cursor_across_pages(monkeypatch):
    urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if "cursor" in request.url.params:
            return httpx.Response(200, json=_page(OWN[1:]))
        return httpx.Response(200, json=_page(OWN[:1], cursor="c1.next"))

    _patch_transport(monkeypatch, handler)
    assert voice_list.fetch_voices("k") == OWN
    assert urls[1].endswith("/v1/voices?limit=100&cursor=c1.next")


def test_fetch_stops_when_has_more_is_false_even_with_a_cursor(monkeypatch):
    handler, _ = _respond(200, json={"voices": OWN, "has_more": False, "next_cursor": "c"})
    _patch_transport(monkeypatch, handler)
    assert voice_list.fetch_voices("k") == OWN


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_key_is_not_authenticated(monkeypatch, status):
    handler, _ = _respond(status, json={"error": "nope"})
    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated):
        voice_list.fetch_voices("bad")


def test_http_error_carries_status_and_body(monkeypatch):
    handler, _ = _respond(500, text="upstream exploded")
    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match=r"\(500\): upstream exploded"):
        voice_list.fetch_voices("k")


def test_http_error_with_empty_body_names_the_status(monkeypatch):
    handler, _ = _respond(400, text="")
    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match=r"\(400\): HTTP 400"):
        voice_list.fetch_voices("k")


def test_transport_failure_is_an_api_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match="Could not reach the TTS service: boom"):
        voice_list.fetch_voices("k")


def test_non_json_body_is_an_api_error(monkeypatch):
    handler, _ = _respond(200, text="<html>")
    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match="not valid JSON"):
        voice_list.fetch_voices("k")


@pytest.mark.parametrize(
    "body",
    [
        {"items": OWN},  # an envelope under the wrong key
        {"voices": "jane"},  # the right key, not a list
        {"voices": ["jane", "michael"]},  # a list, but of strings
        OWN,  # a bare array, not the documented envelope
        "jane",
    ],
)
def test_wrong_shape_is_an_api_error(monkeypatch, body):
    handler, _ = _respond(200, json=body)
    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match="unexpected shape"):
        voice_list.fetch_voices("k")


def test_builtin_voices_are_the_catalog_as_records():
    listing = voice_list.builtin_voices()
    assert [voice["slug"] for voice in listing] == [f"aai-{name}" for name in voices.VOICES]
    assert listing[5] == {
        "slug": "aai-jane",
        "name": "Jane",
        "description": "built-in, speaks every language",
    }


def test_list_voices_is_builtins_then_the_account_voices(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(voice_list, "fetch_voices", lambda key: seen.append(key) or OWN)
    assert voice_list.list_voices("k") == voice_list.builtin_voices() + OWN
    assert seen == ["k"]


def test_format_pads_slugs_and_joins_the_known_details():
    listing: list[dict[str, object]] = [
        {"slug": "aai-jane", "name": "Jane", "language": "english", "description": "warm"},
        {"slug": "bea-12ab34", "language": "spanish", "extra": "ignored"},
    ]
    # Slugs are left-aligned to the widest one; details join with " · " in a fixed
    # order (name, language, description), skipping absent/empty fields and
    # unknown keys. Trailing whitespace is trimmed on a details-less line.
    assert voice_list.format_voice_list(listing) == (
        "aai-jane    Jane · english · warm\nbea-12ab34  spanish"
    )


def test_format_without_details_or_slug_is_still_one_line_per_voice():
    # A record missing its slug renders a "?" placeholder; an empty detail is skipped.
    assert voice_list.format_voice_list([{"slug": "aai-x"}, {"name": ""}]) == "aai-x\n?"


def test_format_empty_listing():
    assert voice_list.format_voice_list([]) == "No voices available."
