"""The streaming-TTS voice listing boundary (aai_cli/tts/voice_list.py): the
GET /v1/voices request shape, error normalization, response parsing, and the
human rendering."""

from __future__ import annotations

import httpx2 as httpx
import pytest

from aai_cli.core import environments
from aai_cli.core.errors import APIError, NotAuthenticated
from aai_cli.tts import voice_list

VOICES = [
    {"voice_id": "aai-jane", "name": "Jane", "language": "en"},
    {"voice_id": "aai-giovanni", "name": "Giovanni", "language": "it"},
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


def test_voices_url_is_the_active_tts_host():
    assert voice_list.voices_url() == (
        "https://streaming-tts.sandbox000.assemblyai-labs.com/v1/voices"
    )


def test_fetch_gets_the_listing_with_the_raw_api_key(monkeypatch):
    handler, seen = _respond(200, json=VOICES)
    client_kwargs = _patch_transport(monkeypatch, handler)
    assert voice_list.fetch_voices("sk_voices") == VOICES
    assert seen["method"] == "GET"
    assert seen["url"] == "https://streaming-tts.sandbox000.assemblyai-labs.com/v1/voices"
    # Streaming TTS takes the raw key, not a Bearer token — same as its socket.
    assert seen["auth"] == "sk_voices"
    assert client_kwargs["timeout"] == 15.0


def test_fetch_unwraps_an_object_envelope(monkeypatch):
    # A {"voices": [...]} envelope yields the same list as a bare array.
    handler, _ = _respond(200, json={"voices": VOICES})
    _patch_transport(monkeypatch, handler)
    assert voice_list.fetch_voices("k") == VOICES


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
        {"items": VOICES},  # an envelope under the wrong key
        {"voices": "jane"},  # the right key, not a list
        ["jane", "michael"],  # a list, but of strings
        "jane",
    ],
)
def test_wrong_shape_is_an_api_error(monkeypatch, body):
    handler, _ = _respond(200, json=body)
    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError, match="unexpected shape"):
        voice_list.fetch_voices("k")


def test_format_pads_ids_and_joins_the_known_details():
    listing: list[dict[str, object]] = [
        {"voice_id": "aai-jane", "name": "Jane", "language": "en", "description": "warm"},
        {"voice_id": "aai-giovanni", "language": "it", "extra": "ignored"},
    ]
    # Ids are left-aligned to the widest one; details join with " · " in a fixed
    # order (name, language, description), skipping absent/empty fields and
    # unknown keys. Trailing whitespace is trimmed on a details-less line.
    assert voice_list.format_voice_list(listing) == (
        "aai-jane      Jane · en · warm\naai-giovanni  it"
    )


def test_format_without_details_or_id_is_still_one_line_per_voice():
    # A record missing its voice_id renders a "?" placeholder; an empty detail is skipped.
    assert voice_list.format_voice_list([{"voice_id": "aai-x"}, {"name": ""}]) == "aai-x\n?"


def test_format_empty_listing():
    assert voice_list.format_voice_list([]) == "No voices available."
