"""HTTP boundary for the Sync STT API (sync.assemblyai.com).

One POST per utterance — multipart audio + JSON config in, the transcript back in
the response body. No polling, no session management; the Universal-3 Pro sync
model handles audio from 80 ms up to 120 s. Backs `assembly dictate`. Like
client.py/ams.py, this module normalizes every failure into the CLIError
hierarchy and stays Rich-free (import-linter contract 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx2 as httpx

from aai_cli import environments, jsonshape
from aai_cli.errors import APIError, auth_failure

# The X-AAI-Model header is required on every Sync API request.
SYNC_MODEL = "u3-sync-pro"
# Documented audio constraints; callers gate on these before uploading.
MIN_AUDIO_MS = 80
MAX_AUDIO_SECONDS = 120
# The request itself has a 30 s inference deadline server-side; leave headroom
# for the upload of up to 40 MB of audio.
_TIMEOUT = 60.0
_RATE_LIMIT_STATUSES = (429, 503)
_HTTP_ERROR_MIN_STATUS = 400


@dataclass(frozen=True)
class SyncTranscript:
    """The fields of a Sync API response the CLI renders."""

    text: str
    confidence: float | None
    audio_duration_ms: int | None
    session_id: str | None


def _detail(resp: httpx.Response) -> str:
    """A human-readable failure detail: the body's message/detail (with its
    error_code when present), else the raw body, else the bare HTTP status."""
    fallback = resp.text or f"HTTP {resp.status_code}"
    try:
        body: object = resp.json()
    except ValueError:
        return fallback
    mapping = jsonshape.as_mapping(body)
    if mapping is None:
        return fallback
    message = mapping.get("message") or mapping.get("detail")
    if message is None:
        return fallback
    code = mapping.get("error_code")
    return f"{message} ({code})" if code else str(message)


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise auth_failure()
    if resp.status_code in _RATE_LIMIT_STATUSES:
        # 429 (rate limit) and 503 (capacity / model cold start) are both
        # transient: the documented recovery is simply to retry shortly.
        raise APIError(
            f"The Sync API is busy ({resp.status_code}): {_detail(resp)}",
            suggestion="Wait a moment and try again.",
        )
    if resp.status_code >= _HTTP_ERROR_MIN_STATUS:
        raise APIError(f"Sync transcription failed ({resp.status_code}): {_detail(resp)}")


def _parse(resp: httpx.Response) -> SyncTranscript:
    try:
        body: object = resp.json()
    except ValueError as exc:
        raise APIError(
            f"The Sync API returned a response that is not valid JSON (HTTP {resp.status_code})."
        ) from exc
    mapping = jsonshape.as_mapping(body)
    if mapping is None or "text" not in mapping:
        raise APIError("The Sync API returned an unexpected response shape (no transcript text).")
    confidence = mapping.get("confidence")
    duration = mapping.get("audio_duration_ms")
    session_id = mapping.get("session_id")
    return SyncTranscript(
        text=str(mapping["text"]),
        confidence=jsonshape.as_float(confidence) if confidence is not None else None,
        audio_duration_ms=jsonshape.as_int(duration) if duration is not None else None,
        session_id=str(session_id) if session_id is not None else None,
    )


def _config(
    sample_rate: int,
    channels: int,
    language_code: str | list[str] | None,
    prompt: str | None,
    word_boost: list[str] | None,
) -> dict[str, object]:
    """The JSON config part: PCM geometry always, model knobs only when set."""
    cfg: dict[str, object] = {"sample_rate": sample_rate, "channels": channels}
    if language_code:
        cfg["language_code"] = language_code
    if prompt:
        cfg["prompt"] = prompt
    if word_boost:
        cfg["word_boost"] = list(word_boost)
    return cfg


def transcribe_pcm(
    api_key: str,
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    language_code: str | list[str] | None = None,
    prompt: str | None = None,
    word_boost: list[str] | None = None,
) -> SyncTranscript:
    """POST raw PCM (S16LE) to the Sync API and return the finished transcript."""
    cfg = _config(sample_rate, channels, language_code, prompt, word_boost)
    files: dict[str, tuple[str | None, bytes | str, str | None]] = {
        "audio": ("audio.pcm", pcm, "audio/pcm"),
        "config": (None, json.dumps(cfg), "application/json"),
    }
    headers = {"authorization": api_key, "x-aai-model": SYNC_MODEL}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{environments.active().sync_base}/transcribe", headers=headers, files=files
            )
    except httpx.HTTPError as exc:
        raise APIError(
            f"Could not reach the Sync API: {exc}",
            suggestion="Check your network connection and try again.",
        ) from exc
    _raise_for_error(resp)
    return _parse(resp)
