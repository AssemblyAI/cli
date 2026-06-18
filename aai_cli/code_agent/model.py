"""Build the agent's chat model — always the AssemblyAI LLM Gateway.

The gateway is OpenAI-compatible, so we reach it through ``langchain_openai.ChatOpenAI``
pointed at the active environment's gateway base. This is the *only* model wiring the
coding agent has: there is no path to a third-party provider, so a coding session can
never silently send the user's code to anything but AssemblyAI.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from aai_cli.core import environments

# The gateway omits Anthropic's required ``tool_use.input`` when an OpenAI tool call's
# ``arguments`` is empty (``""`` / ``"{}"``); substitute a minimal non-empty object so the
# field is emitted. See :func:`_ensure_tool_call_arguments`.
_PLACEHOLDER_ARGUMENTS = '{"_": ""}'

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGenerationChunk


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


def _hoist_tool_call_ids(chunk: object) -> None:
    """Normalize a streamed chunk's tool-call deltas: drop blank ones, hoist nested ids.

    Two AssemblyAI LLM Gateway streaming quirks, both fixed in place before langchain
    converts the chunk:

    1. **Spurious blank deltas.** Every streamed turn (when tools are available) starts with
       an empty tool-call delta — ``{"function": {"id": "", "name": "", "arguments": ""}}``.
       On a pure-text turn no real call follows, so langchain is left with a tool call whose
       ``name`` is ``""``; deepagents then dispatches it and the turn dies with
       ``Error:  is not a valid tool``. We drop any delta with no name, id, or arguments
       (which also harmlessly drops the gateway's empty argument-continuation deltas).
    2. **Misplaced id.** The id is nested under ``function`` instead of at the tool-call top
       level where the OpenAI spec and ``langchain_openai`` (``id=rtc.get("id")``) read it,
       so without help every call parses with ``id=None`` and its reply ``ToolMessage`` fails
       validation. We move it back up; the id rides only a call's first delta.

    (The non-streaming endpoint has neither quirk, so only the streaming path needs this.)
    """
    if not isinstance(chunk, dict):
        return
    choices = chunk.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            _hoist_in_choice(choice)


def _hoist_in_choice(choice: object) -> None:
    """Drop blank tool-call deltas, then hoist ids, within one streamed choice's delta."""
    if not isinstance(choice, dict):
        return
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        delta["tool_calls"] = [tc for tc in tool_calls if not _is_blank_tool_call(tc)]
        _hoist_call_list(delta["tool_calls"])


def _is_blank_tool_call(tool_call: object) -> bool:
    """True for the gateway's spurious empty tool-call delta (no name, id, or arguments)."""
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return False
    return not function.get("name") and not function.get("id") and not function.get("arguments")


def _hoist_call_list(tool_calls: list[object]) -> None:
    """Hoist a misplaced ``function.id`` to the tool-call top level for each call in the list.

    Helper for :func:`_hoist_tool_call_ids` — split out so the per-chunk traversal stays
    under the complexity bar. A call is rewritten only when it carries an ``id`` nested
    under ``function`` (the gateway's misplaced first-delta shape). This stays idempotent
    once the gateway is fixed: a correct delta puts the id at the top level and leaves no
    ``function.id``, so the move never fires.
    """
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("id") is not None:
            tool_call["id"] = function.pop("id")


def _ensure_tool_call_arguments(messages: object) -> None:
    """Give every empty tool-call ``arguments`` a non-empty placeholder object, in place.

    The AssemblyAI LLM Gateway maps each OpenAI tool call's ``arguments`` (a JSON string)
    onto Anthropic's ``tool_use.input`` object, but drops ``input`` entirely when the
    arguments are empty (``""`` or ``"{}"``). Anthropic *requires* ``input`` to be present,
    so replaying any argument-less tool call is rejected (400, surfaced as a 500 while
    streaming) — and because the failing call sits in the conversation history, every later
    turn fails too, wedging the session. We swap in a minimal non-empty object so the gateway
    emits a valid ``input``. This only rewrites the request we send: the tool already ran
    locally with its real (empty) arguments, and the gateway accepts the placeholder even for
    tools that declare ``additionalProperties: false``. (Drop this once the gateway maps empty
    arguments to ``input: {}`` itself.)
    """
    if not isinstance(messages, list):
        return
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(tool_calls, list):
            _fill_empty_arguments(tool_calls)


def _fill_empty_arguments(tool_calls: list[object]) -> None:
    """Replace each empty ``function.arguments`` with the placeholder (helper for the above)."""
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and _is_empty_arguments(function.get("arguments")):
            function["arguments"] = _PLACEHOLDER_ARGUMENTS


def _is_empty_arguments(arguments: object) -> bool:
    """True when ``arguments`` is an OpenAI args string carrying no fields (``""``/``"{}"``)."""
    if not isinstance(arguments, str):
        return False
    stripped = arguments.strip()
    if not stripped:
        return True
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return False
    return isinstance(parsed, dict) and not parsed


def build_model(
    api_key: str,
    *,
    model: str,
    max_tokens: int | None = None,
    extra: Mapping[str, object] | None = None,
) -> BaseChatModel:
    """A ChatOpenAI bound to the active environment's LLM Gateway.

    ``use_responses_api=False`` keeps it on the chat-completions endpoint the gateway
    implements (the same one `aai_cli.core.llm` uses), rather than the OpenAI
    Responses API that langchain would otherwise prefer for ``openai:`` models. The
    subclass also flattens content-parts arrays the gateway rejects (see
    :func:`_flatten_content`) and repairs misplaced streamed tool-call ids (see
    :func:`_hoist_tool_call_ids`).

    ``max_tokens`` caps the per-reply length (the live voice agent passes a small cap to
    keep spoken replies short and fast); ``extra`` passes any additional gateway request
    fields through as ``extra_body`` (so they reach the request body verbatim, like
    `aai_cli.core.llm`'s ``extra``). Both default to off so the coding agent's call is
    unchanged.
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    class _GatewayChatOpenAI(ChatOpenAI):
        """ChatOpenAI that adapts the gateway's OpenAI-incompatible quirks for langchain.

        Three fix-ups, each working around a gateway request/response bug the upstream client
        doesn't expect: flatten list-content messages the gateway 500s on and give empty
        tool-call arguments a placeholder the gateway can map to ``tool_use.input`` (request
        side, see :func:`_flatten_content` / :func:`_ensure_tool_call_arguments`), and hoist
        each streamed tool-call ``id`` back to the tool-call top level where langchain reads it
        (response side, see :func:`_hoist_tool_call_ids`).
        """

        def _get_request_payload(
            self, input_: object, *, stop: list[str] | None = None, **kwargs: object
        ) -> dict:
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            messages = payload.get("messages")
            _flatten_content(messages)
            _ensure_tool_call_arguments(messages)
            return payload

        def _convert_chunk_to_generation_chunk(
            self,
            chunk: dict,
            default_chunk_class: type,
            base_generation_info: dict | None,
        ) -> ChatGenerationChunk | None:
            _hoist_tool_call_ids(chunk)
            return super()._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, base_generation_info
            )

    return _GatewayChatOpenAI(
        model=model,
        base_url=environments.active().llm_gateway_base,
        api_key=SecretStr(api_key),
        use_responses_api=False,
        max_tokens=max_tokens,
        extra_body=dict(extra) if extra else None,
    )
