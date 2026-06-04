from __future__ import annotations

from typing import Any

import openai
from openai import OpenAI

from assemblyai_cli.errors import APIError, auth_failure

# The LLM Gateway is OpenAI-compatible, so we talk to it through the OpenAI SDK
# pointed at this base URL. (The synchronous gateway has no assemblyai-SDK client.)
GATEWAY_BASE_URL = "https://llm-gateway.assemblyai.com/v1"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1000

# Exact tag the gateway substitutes with a transcript's text when `transcript_id`
# is supplied. Must be exactly "{{ transcript }}" (spaces included).
TRANSCRIPT_TAG = "{{ transcript }}"

# A curated subset for `aai llm --list-models` and help text. The gateway is the
# source of truth for what's actually accepted, so we don't validate against this.
KNOWN_MODELS = (
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "gpt-5.1",
    "gpt-5",
    "gpt-4.1",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


def build_messages(
    prompt: str,
    *,
    system: str | None = None,
    transcript_id: str | None = None,
    transcript_text: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat `messages` array for a transcript transform or plain prompt.

    With a `transcript_id`, the gateway injects the transcript server-side, so we
    append the `{{ transcript }}` tag. Otherwise any `transcript_text` is inlined.
    """
    if transcript_id is not None:
        content = f"{prompt}\n\n{TRANSCRIPT_TAG}"
    elif transcript_text is not None:
        content = f"{prompt}\n\nTranscript:\n{transcript_text}"
    else:
        content = prompt
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL)


def complete(
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    transcript_id: str | None = None,
) -> Any:
    """Create a chat completion via the gateway and return the OpenAI response.

    `transcript_id` is passed through as an extra body field so the gateway can
    inject the transcript text server-side. Auth failures map to NotAuthenticated
    and everything else to APIError, matching the rest of the CLI.
    """
    client = _client(api_key)
    extra_body = {"transcript_id": transcript_id} if transcript_id is not None else None
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        raise auth_failure() from exc
    except openai.OpenAIError as exc:
        raise APIError(f"LLM Gateway request failed: {exc}") from exc


def content_of(response: Any) -> str:
    """Pull the assistant's text out of a chat-completions response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise APIError("LLM Gateway response contained no message content.") from exc
    return content or ""


def usage_of(response: Any) -> dict[str, Any] | None:
    """Return the token-usage block as a plain dict, if present."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        dumped: dict[str, Any] = usage.model_dump()
        return dumped
    if isinstance(usage, dict):
        return usage
    return None


def transform_transcript(
    api_key: str,
    *,
    prompt: str,
    model: str = DEFAULT_MODEL,
    transcript_id: str | None = None,
    transcript_text: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run `prompt` over a transcript (by id or inline text) and return the result."""
    messages = build_messages(prompt, transcript_id=transcript_id, transcript_text=transcript_text)
    response = complete(
        api_key,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        transcript_id=transcript_id,
    )
    return content_of(response)
