from __future__ import annotations

from typing import Any

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from aai_cli import environments
from aai_cli.errors import APIError

# The LLM Gateway is OpenAI-compatible, so we talk to it through the OpenAI SDK
# pointed at this base URL. This is the production host used in generated code
# snippets (code_gen); runtime calls use the active environment's gateway base.
GATEWAY_BASE_URL = "https://llm-gateway.assemblyai.com/v1"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
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
    return OpenAI(api_key=api_key, base_url=environments.active().llm_gateway_base)


def complete(
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    transcript_id: str | None = None,
) -> ChatCompletion:
    """Create a chat completion via the gateway and return the OpenAI response.

    `transcript_id` is passed through as an extra body field so the gateway can
    inject the transcript text server-side. Access/permission and other gateway
    errors surface the gateway's own message as APIError.
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
        # The gateway returns 401/403 for both an invalid key and a plan
        # entitlement block ("no access to LLM Gateway"), so surface its actual
        # message rather than a generic "run aai login" that misleads unpaid
        # accounts (the key is fine; the feature requires a paid plan).
        raise APIError(f"LLM Gateway access denied: {exc}") from exc
    except openai.OpenAIError as exc:
        raise APIError(f"LLM Gateway request failed: {exc}") from exc


def content_of(response: ChatCompletion) -> str:
    """Pull the assistant's text out of a chat-completions response."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise APIError("LLM Gateway response contained no message content.") from exc
    return content or ""


def usage_of(response: ChatCompletion) -> dict[str, Any] | None:
    """Return the token-usage block as a plain dict, if present."""
    usage = response.usage
    if usage is None:
        return None
    dumped: dict[str, Any] = usage.model_dump()
    return dumped


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


def run_chain(
    api_key: str,
    prompts: list[str],
    *,
    transcript_text: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run a chain of prompts over inline transcript text and return the final output.

    The first prompt runs over `transcript_text`; each subsequent prompt runs over the
    previous prompt's response. Used by live streaming (`stream --llm`), where there is
    no transcript id to inject server-side, so the text is always inlined.
    """
    output = ""
    text = transcript_text
    for prompt in prompts:
        output = transform_transcript(
            api_key, prompt=prompt, model=model, max_tokens=max_tokens, transcript_text=text
        )
        text = output
    return output
