"""HTTP boundary for the Dictation API (dictation.assemblyai.com).

One POST per utterance to ``/v1/transcribe/live`` (/docs/dictation): a multipart
body whose JSON ``config`` part comes first and whose ``audio`` part (raw 16-bit
PCM, at most 120 s) follows, and the result back in the response — the verbatim
transcript in ``text`` and the send-ready cleanup rewrite in ``llm_response``.
Backs `assembly dictate`. Like client.py/ams.py, this module normalizes every
failure into the CLIError hierarchy and stays Rich-free (import-linter contract 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx2 as httpx

from aai_cli.core import environments, jsonshape
from aai_cli.core.errors import APIError, auth_failure

# The live endpoint's documented cap on one utterance.
MAX_AUDIO_SECONDS = 120
# Generous enough for a full 120 s upload plus transcription and the rewrite.
_TIMEOUT = 90.0
_RETRY_STATUSES = (429, 502, 503, 504)
_HTTP_ERROR_MIN_STATUS = 400


@dataclass(frozen=True)
class DictationResult:
    """The fields of a Dictation response the CLI renders."""

    text: str
    llm_response: str | None
    llm_error: str | None
    confidence: float | None
    audio_duration_ms: int | None
    session_id: str | None

    @property
    def final_text(self) -> str:
        """The rewrite, falling back to the verbatim transcript when it failed."""
        return self.llm_response if self.llm_response is not None else self.text


# Import-time line: coverage credits whichever test imported this module first, so the
# mutation gate can't route it to test_config_is_immutable, which pins it.
@dataclass(frozen=True)  # pragma: no mutate
class DictationConfig:
    """The optional transcription and rewrite knobs; unset ones are left out."""

    language_codes: list[str] | None = None
    stt_prompt: str | None = None
    keyterms: list[str] | None = None
    llm_instruction: str | None = None
    rewrite_mode: str | None = None

    def body(self, *, sample_rate: int, channels: int) -> dict[str, object]:
        """The JSON ``config`` part: PCM geometry always, the knobs only when set."""
        cfg: dict[str, object] = {"sample_rate": sample_rate, "channels": channels}
        if self.language_codes:
            cfg["language_codes"] = list(self.language_codes)
        if self.stt_prompt:
            cfg["stt_prompt"] = self.stt_prompt
        if self.keyterms:
            cfg["keyterms_prompt"] = list(self.keyterms)
        if self.llm_instruction:
            cfg["llm_instruction"] = self.llm_instruction
        if self.rewrite_mode:
            cfg["rewrite_mode"] = self.rewrite_mode
        return cfg


def _detail(resp: httpx.Response) -> str:
    """A human-readable failure detail. Errors come in two shapes — ``detail`` (auth,
    rate limits, transcription) or ``error`` + ``error_code`` (Dictation's own) — so
    read ``detail`` first, then ``error``, else the raw body or the bare status."""
    fallback = resp.text or f"HTTP {resp.status_code}"
    try:
        body: object = resp.json()
    except ValueError:
        return fallback
    mapping = jsonshape.as_mapping(body)
    if mapping is None:
        return fallback
    message = mapping.get("detail") or mapping.get("error")
    if message is None:
        return fallback
    code = mapping.get("error_code")
    return f"{message} ({code})" if code else str(message)


def _raise_for_error(resp: httpx.Response) -> None:
    if resp.status_code in (401, 403):
        raise auth_failure()
    if resp.status_code in _RETRY_STATUSES:
        # Rate limit, capacity, and upstream failures/timeouts: the documented
        # recovery is to back off and retry.
        raise APIError(
            f"The Dictation API is busy ({resp.status_code}): {_detail(resp)}",
            suggestion="Wait a moment and try again.",
        )
    if resp.status_code >= _HTTP_ERROR_MIN_STATUS:
        raise APIError(f"Dictation failed ({resp.status_code}): {_detail(resp)}")


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _parse(resp: httpx.Response) -> DictationResult:
    try:
        body: object = resp.json()
    except ValueError as exc:
        raise APIError(
            f"The Dictation API returned a response that is not valid JSON "
            f"(HTTP {resp.status_code})."
        ) from exc
    mapping = jsonshape.as_mapping(body)
    if mapping is None or "text" not in mapping:
        raise APIError("The Dictation API returned an unexpected response shape (no text).")
    confidence = mapping.get("confidence")
    duration = mapping.get("audio_duration_ms")
    return DictationResult(
        text=str(mapping["text"]),
        llm_response=_optional_str(mapping.get("llm_response")),
        llm_error=_optional_str(mapping.get("llm_error")),
        confidence=jsonshape.as_float(confidence) if confidence is not None else None,
        audio_duration_ms=jsonshape.as_int(duration) if duration is not None else None,
        session_id=_optional_str(mapping.get("session_id")),
    )


def transcribe_live(
    api_key: str,
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int = 1,
    config: DictationConfig | None = None,
) -> DictationResult:
    """POST raw PCM (S16LE) to ``/v1/transcribe/live`` and return the result."""
    cfg = (config or DictationConfig()).body(sample_rate=sample_rate, channels=channels)
    # Order matters: the server starts transcribing as the body arrives and can't
    # without the config, so `config` must precede `audio` (else a 400).
    files: dict[str, tuple[str | None, bytes | str, str | None]] = {
        "config": (None, json.dumps(cfg), "application/json"),
        "audio": ("audio.pcm", pcm, "audio/pcm"),
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{environments.active().dictation_base}/v1/transcribe/live",
                headers={"authorization": api_key},
                files=files,
            )
    except httpx.HTTPError as exc:
        raise APIError(
            f"Could not reach the Dictation API: {exc}",
            suggestion="Check your network connection and try again.",
        ) from exc
    _raise_for_error(resp)
    return _parse(resp)
