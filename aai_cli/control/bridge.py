"""Adapt the LLM Gateway into the engine's :data:`~aai_cli.control.engine.Responder`.

The gateway is OpenAI-compatible, so one chat-completions call with the control
``tools`` is a single model turn. This converts the SDK response into the
engine's plain :class:`~aai_cli.control.engine.Reply` — parsing each tool call's
JSON arguments — so the loop never touches the OpenAI types. The underlying
:func:`aai_cli.core.llm.complete` is injected so the adapter is unit-tested
against a fake completer with no network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from aai_cli.control import engine, tools
from aai_cli.control.engine import Reply, ToolCall
from aai_cli.core import jsonshape, llm

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from aai_cli.control.engine import Message

# The completer seam: same shape as ``llm.complete``'s keyword call below.
type Completer = Callable[..., ChatCompletion]


def _parse_arguments(raw: str | None) -> dict[str, object]:
    """Parse a tool call's JSON ``arguments`` string into a dict.

    A model occasionally emits empty or malformed arguments; treat those as no
    arguments so validation (not a JSON crash) reports the real problem.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return jsonshape.as_mapping(parsed) or {}


def _reply_of(response: ChatCompletion) -> Reply:
    """Convert a chat-completions response into the engine's :class:`Reply`."""
    message = response.choices[0].message
    calls: list[ToolCall] = []
    for call in message.tool_calls or []:
        # The SDK union also allows a custom (non-function) tool call; we only ask
        # the model for function tools, so narrow to those on the type discriminant.
        if call.type != "function":
            continue
        calls.append(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
        )
    return Reply(content=message.content or "", tool_calls=tuple(calls))


def build_responder(
    api_key: str,
    *,
    model: str,
    max_tokens: int,
    complete: Completer = llm.complete,
) -> engine.Responder:
    """A :data:`Responder` that runs one gateway turn with the control tools.

    The tools and ``tool_choice`` ride in ``extra`` (merged into the request
    body), since the gateway accepts the OpenAI tool-calling fields.
    """

    def respond(messages: list[Message]) -> Reply:
        response = complete(
            api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra={"tools": tools.tool_definitions(), "tool_choice": "auto"},
        )
        return _reply_of(response)

    return respond
