from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from aai_cli.core import environments
from aai_cli.core.errors import APIError, UsageError, auth_failure

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

# The LLM Gateway is OpenAI-compatible, so we talk to it through the OpenAI SDK
# pointed at the active environment's gateway base (see _client / code_gen).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1000

# Exact tag the gateway substitutes with a transcript's text when `transcript_id`
# is supplied. Must be exactly "{{ transcript }}" (spaces included).
TRANSCRIPT_TAG = "{{ transcript }}"

# A curated subset for `assembly llm --list-models` and help text. The gateway is the
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


def complete_model(incomplete: str) -> list[str]:
    """Shell-completion callback for ``--model``: known model ids matching the prefix.

    The gateway accepts more than this curated list, so completion only *suggests*
    these — it never restricts what you can type.
    """
    return [m for m in KNOWN_MODELS if m.startswith(incomplete)]


def parse_gateway_overrides(pairs: Sequence[str]) -> dict[str, object]:
    """``--config KEY=VALUE`` pairs to typed gateway request fields.

    The escape hatch for request fields the curated flags don't cover (the same
    role ``--config`` plays on `transcribe`/`stream`). The gateway's field set is
    open-ended (it is OpenAI-compatible per model family), so values aren't
    allow-listed; each VALUE parses as JSON when it can (``temperature=0.2`` →
    float, ``stop=["END"]`` → list) and falls back to the literal string
    otherwise (``reasoning_effort=low``).
    """
    extra: dict[str, object] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise UsageError(
                f"--config expects KEY=VALUE, got {pair!r}.",
                suggestion="e.g. --config temperature=0.2",
            )
        try:
            extra[key] = json.loads(raw)
        except ValueError:
            extra[key] = raw
    return extra


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
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=environments.active().llm_gateway_base)


# Lowercased substrings that mark a gateway 401/403 as a plan-entitlement block
# rather than a bad key or an intercepting proxy. "no access" is the gateway's own
# phrasing for accounts without the LLM Gateway entitlement.
_ENTITLEMENT_HINTS = ("entitle", "plan", "upgrade", "billing", "no access")

_PAID_PLAN_SUGGESTION = (
    "The LLM Gateway requires a paid plan — check your plan at "
    "https://www.assemblyai.com/dashboard."
)
_ACCESS_DENIED_SUGGESTION = (
    "Check your API key ('assembly login') and that your network/proxy allows the "
    "LLM Gateway, then try again."
)


def _is_entitlement_denial(exc: object) -> bool:
    """True when a gateway 401/403 reads as a plan-entitlement block rather than a
    bad key or an intercepting proxy."""
    text = f"{exc} {getattr(exc, 'body', None) or ''}".lower()
    return any(hint in text for hint in _ENTITLEMENT_HINTS)


def _denial_suggestion(exc: object) -> str:
    """Pick the suggestion for a gateway 401/403: point at billing only when the
    response actually mentions the plan entitlement, otherwise at key/network —
    a corporate-proxy 403 must not send users to the billing page."""
    if _is_entitlement_denial(exc):
        return _PAID_PLAN_SUGGESTION
    return _ACCESS_DENIED_SUGGESTION


def complete(
    api_key: str,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    transcript_id: str | None = None,
    extra: dict[str, object] | None = None,
) -> ChatCompletion:
    """Create a chat completion via the gateway and return the OpenAI response.

    `transcript_id` is passed through as an extra body field so the gateway can
    inject the transcript text server-side; `extra` carries the user's ``--config``
    overrides the same way. Access/permission and other gateway errors surface
    the gateway's own message as APIError.
    """
    import openai

    client = _client(api_key)
    extra_body: dict[str, object] = {}
    if transcript_id is not None:
        extra_body["transcript_id"] = transcript_id
    if extra:
        extra_body.update(extra)
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            extra_body=extra_body or None,
        )
    except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
        # The gateway returns 401/403 for an invalid key, a proxy block, and a
        # plan entitlement block ("no access to LLM Gateway"). A plain 401
        # (AuthenticationError) with no entitlement hint is just a rejected key, so
        # surface the same clean exit-4 auth_failure transcribe gives instead of
        # echoing the gateway's raw 401 body. A 403 (proxy or entitlement) keeps the
        # gateway's own message and picks the suggestion from what it says — only an
        # entitlement message should point at billing, never a corporate-proxy 403.
        if isinstance(exc, openai.AuthenticationError) and not _is_entitlement_denial(exc):
            raise auth_failure() from exc
        raise APIError(
            f"LLM Gateway access denied: {exc}",
            suggestion=_denial_suggestion(exc),
        ) from exc
    except openai.OpenAIError as exc:
        raise APIError(
            f"LLM Gateway request failed: {exc}",
            suggestion="Check your network and try again.",
        ) from exc


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
    return usage.model_dump()


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
    steps = run_chain_steps(
        api_key,
        prompts,
        transcript_text=transcript_text,
        model=model,
        max_tokens=max_tokens,
    )
    return steps[-1]["output"] if steps else ""


def run_chain_steps(
    api_key: str,
    prompts: list[str],
    *,
    transcript_id: str | None = None,
    transcript_text: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, str]]:
    """Run a prompt chain and return each step's prompt/output pair.

    The first step runs over a server-injected transcript when `transcript_id` is
    provided, otherwise over inline `transcript_text`. Later steps run over the
    previous step's output.
    """
    if not prompts:
        return []

    # Exactly one of transcript_id / transcript_text is set by callers; pass both
    # through (build_messages prefers the id) so the two cases share one call.
    output = transform_transcript(
        api_key,
        prompt=prompts[0],
        model=model,
        max_tokens=max_tokens,
        transcript_id=transcript_id,
        transcript_text=transcript_text,
    )
    steps = [{"prompt": prompts[0], "output": output}]

    for prompt in prompts[1:]:
        output = transform_transcript(
            api_key,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            transcript_text=output,
        )
        steps.append({"prompt": prompt, "output": output})

    return steps
