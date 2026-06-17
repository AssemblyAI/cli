"""Build the agent's chat model — always the AssemblyAI LLM Gateway.

The gateway is OpenAI-compatible, so we reach it through ``langchain_openai.ChatOpenAI``
pointed at the active environment's gateway base. This is the *only* model wiring the
coding agent has: there is no path to a third-party provider, so a coding session can
never silently send the user's code to anything but AssemblyAI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aai_cli.core import environments

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def _flatten_content(messages: object) -> None:
    """Collapse any OpenAI 'content-parts' array to a plain string, in place.

    deepagents/langchain serialize the system prompt (and some messages) as a list of
    ``{"type": "text", "text": …}`` blocks. The AssemblyAI LLM Gateway's
    ``/v1/chat/completions`` only accepts plain-string content and returns an opaque 500
    on a content array (unlike `aai_cli.core.llm`, which always sends strings) — so we
    join the text parts back into one string for every message before the request goes out.
    """
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )


def build_model(api_key: str, *, model: str) -> BaseChatModel:
    """A ChatOpenAI bound to the active environment's LLM Gateway.

    ``use_responses_api=False`` keeps it on the chat-completions endpoint the gateway
    implements (the same one `aai_cli.core.llm` uses), rather than the OpenAI
    Responses API that langchain would otherwise prefer for ``openai:`` models. The
    subclass also flattens content-parts arrays the gateway rejects (see
    :func:`_flatten_content`).
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    class _GatewayChatOpenAI(ChatOpenAI):
        """ChatOpenAI that rewrites list-content messages to plain strings for the gateway."""

        def _get_request_payload(
            self, input_: object, *, stop: list[str] | None = None, **kwargs: object
        ) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            _flatten_content(payload.get("messages"))
            return payload

    return _GatewayChatOpenAI(
        model=model,
        base_url=environments.active().llm_gateway_base,
        api_key=SecretStr(api_key),
        use_responses_api=False,
    )
