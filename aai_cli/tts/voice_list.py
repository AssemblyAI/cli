"""The streaming-TTS voice listing behind ``--list-voices`` on `speak`, `dub`, and `live`.

Two sources, per the TTS docs (/docs/tts/voices): the built-in voices (the static
``voices`` catalog — they speak every language) and the account's own voices, which
the service lists at ``GET /v1/voices`` on the TTS host (its default scope; cloned
voices come and go without a CLI release, so they're fetched). Each record's ``slug``
is the ``voice_id`` to pass as ``--voice``. Like ``core/dictation.py``, every failure
is normalized into the CLIError hierarchy and the module stays Rich-free.
"""

from __future__ import annotations

import httpx2 as httpx

from aai_cli.core import environments, jsonshape
from aai_cli.core.errors import APIError, auth_failure
from aai_cli.tts import voices

_TIMEOUT = 15.0
_HTTP_ERROR_MIN_STATUS = 400
# The largest page the listing serves; pages are followed via next_cursor.
_PAGE_LIMIT = 100
# The per-voice fields worth a column in the human listing, when the record has them.
_DETAIL_KEYS = ("name", "language", "description")
_BUILTIN_DESCRIPTION = "built-in, speaks every language"


def voices_url() -> str:
    """The voice-listing endpoint for the active environment's TTS host."""
    return f"https://{environments.active().streaming_tts_host}/v1/voices"


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise auth_failure()
    if resp.status_code >= _HTTP_ERROR_MIN_STATUS:
        detail = resp.text or f"HTTP {resp.status_code}"
        raise APIError(f"Could not list TTS voices ({resp.status_code}): {detail}")


def _parse(resp: httpx.Response) -> tuple[list[dict[str, object]], str | None]:
    """One page: its voice objects (under ``voices``) and the cursor of the next page,
    or None on the last one. Other shapes are a malformed response."""
    try:
        body: object = resp.json()
    except ValueError as exc:
        raise APIError("The TTS service returned a voice list that is not valid JSON.") from exc
    wrapper = jsonshape.as_mapping(body)
    listing = jsonshape.as_object_list(wrapper.get("voices")) if wrapper is not None else None
    if wrapper is None or listing is None:
        raise APIError("The TTS service returned a voice list in an unexpected shape.")
    cursor = wrapper.get("next_cursor")
    has_more = wrapper.get("has_more") is True
    return listing, cursor if has_more and isinstance(cursor, str) else None


def builtin_voices() -> list[dict[str, object]]:
    """The built-in voices in the listing's record shape."""
    return [
        {"slug": voices.voice_id(name), "name": name.title(), "description": _BUILTIN_DESCRIPTION}
        for name in voices.VOICES
    ]


def fetch_voices(api_key: str) -> list[dict[str, object]]:
    """The account's own voices (every page), each a JSON object with its ``slug``."""
    listing: list[dict[str, object]] = []
    params: dict[str, str | int] = {"limit": _PAGE_LIMIT}
    # Streaming TTS takes the raw API key in Authorization, like its socket.
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            while True:
                resp = client.get(voices_url(), params=params, headers={"authorization": api_key})
                _raise_for_error(resp)
                page, cursor = _parse(resp)
                listing.extend(page)
                if cursor is None:
                    return listing
                params = {"limit": _PAGE_LIMIT, "cursor": cursor}
    except httpx.HTTPError as exc:
        raise APIError(
            f"Could not reach the TTS service: {exc}",
            suggestion="Check your network connection and try again.",
        ) from exc


def list_voices(api_key: str) -> list[dict[str, object]]:
    """Every voice ``--voice`` accepts: the built-in voices, then the account's own."""
    return builtin_voices() + fetch_voices(api_key)


def format_voice_list(listing: list[dict[str, object]]) -> str:
    """One line per voice: the ``slug`` (what ``--voice`` accepts), then any
    name/language/description the record carries, so the output stays useful if
    the record shape grows."""
    if not listing:
        return "No voices available."
    slugs = [str(voice.get("slug", "?")) for voice in listing]
    width = max(len(slug) for slug in slugs)
    lines: list[str] = []
    # slugs is derived from listing, so the lengths always match: strict is unobservable.
    for slug, voice in zip(slugs, listing, strict=True):  # pragma: no mutate
        details = " · ".join(str(voice[key]) for key in _DETAIL_KEYS if voice.get(key))
        lines.append(f"{slug:<{width}}  {details}".rstrip())
    return "\n".join(lines)
