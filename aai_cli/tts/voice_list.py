"""The live streaming-TTS voice listing: ``GET /v1/voices`` on the TTS host.

Backs ``--list-voices`` on `speak`, `dub`, and `live`. The service is the source of
truth for which voices exist (library voices come and go without a CLI release),
so the listing is fetched, not read from the static ``voices`` catalog — which
only drives the language-driven defaults and shell completion. Like
``core/sync_stt.py``, every failure is normalized into the CLIError hierarchy and
the module stays Rich-free.
"""

from __future__ import annotations

import httpx2 as httpx

from aai_cli.core import environments, jsonshape
from aai_cli.core.errors import APIError, auth_failure

_TIMEOUT = 15.0
_HTTP_ERROR_MIN_STATUS = 400
# The per-voice fields worth a column in the human listing, when the service sends them.
_DETAIL_KEYS = ("name", "language", "description")


def voices_url() -> str:
    """The voice-listing endpoint for the active environment's TTS host."""
    return f"https://{environments.active().streaming_tts_host}/v1/voices"


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise auth_failure()
    if resp.status_code >= _HTTP_ERROR_MIN_STATUS:
        detail = resp.text or f"HTTP {resp.status_code}"
        raise APIError(f"Could not list TTS voices ({resp.status_code}): {detail}")


def _parse(resp: httpx.Response) -> list[dict[str, object]]:
    """The voice objects out of the response: either a bare JSON array or an object
    wrapping it under ``voices``. Other shapes are a malformed response."""
    try:
        body: object = resp.json()
    except ValueError as exc:
        raise APIError("The TTS service returned a voice list that is not valid JSON.") from exc
    wrapper = jsonshape.as_mapping(body)
    listing = jsonshape.as_object_list(wrapper.get("voices") if wrapper is not None else body)
    if listing is None:
        raise APIError("The TTS service returned a voice list in an unexpected shape.")
    return listing


def fetch_voices(api_key: str) -> list[dict[str, object]]:
    """The service's voices, each a JSON object carrying at least its ``voice_id``."""
    # Streaming TTS authenticates with the raw API key (no Bearer prefix), like its socket.
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(voices_url(), headers={"authorization": api_key})
    except httpx.HTTPError as exc:
        raise APIError(
            f"Could not reach the TTS service: {exc}",
            suggestion="Check your network connection and try again.",
        ) from exc
    _raise_for_error(resp)
    return _parse(resp)


def format_voice_list(listing: list[dict[str, object]]) -> str:
    """One line per voice: the ``voice_id`` (what ``--voice`` accepts), then any
    name/language/description the service sent, so the output stays useful if
    the record shape grows."""
    if not listing:
        return "No voices available."
    ids = [str(voice.get("voice_id", "?")) for voice in listing]
    width = max(len(voice_id) for voice_id in ids)
    lines: list[str] = []
    for voice_id, voice in zip(ids, listing, strict=True):
        details = " · ".join(str(voice[key]) for key in _DETAIL_KEYS if voice.get(key))
        lines.append(f"{voice_id:<{width}}  {details}".rstrip())
    return "\n".join(lines)
