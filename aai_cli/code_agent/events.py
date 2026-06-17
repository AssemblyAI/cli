"""Turn the agent's langchain messages into framework-agnostic display events.

Both the Rich renderer and the Textual TUI consume the same small event vocabulary,
so the message-shape knowledge (AIMessage tool_calls, ToolMessage results) lives here
once rather than in each front-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class AssistantText:
    """A chunk of the assistant's natural-language reply."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """The agent's request to run a tool (announced when not gated by approval)."""

    name: str
    args: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """A tool's returned output, surfaced back into the conversation."""

    name: str
    content: str


@dataclass(frozen=True)
class ErrorText:
    """A turn failed (e.g. the gateway errored); shown instead of crashing the UI."""

    text: str


Event = AssistantText | ToolCall | ToolResult | ErrorText


def _text_of(content: object) -> str:
    """Coerce a message's content (str, or a list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in content
        ]
        return "".join(parts)
    return str(content)


def message_events(message: BaseMessage, *, announce_calls: bool) -> list[Event]:
    """Display events for one new message.

    Assistant text always shows; tool calls show only when ``announce_calls`` (the
    --auto path, where no approval prompt announced them); tool results always show.
    A human message produces nothing — the UI already echoed the user's own input.
    """
    kind = type(message).__name__
    if kind == "ToolMessage":
        return [
            ToolResult(
                name=getattr(message, "name", "") or "tool", content=_text_of(message.content)
            )
        ]
    if kind == "AIMessage":
        events: list[Event] = []
        text = _text_of(message.content).strip()
        if text:
            events.append(AssistantText(text))
        if announce_calls:
            events.extend(
                ToolCall(name=call.get("name", ""), args=call.get("args", {}))
                for call in getattr(message, "tool_calls", None) or []
            )
        return events
    return []


def new_messages(result: dict[str, object], already_seen: int) -> list[BaseMessage]:
    """The messages added to the conversation since ``already_seen`` were rendered."""
    messages = result.get("messages")
    if not isinstance(messages, list):
        return []
    return messages[already_seen:]


def interrupt_request(result: dict[str, object]) -> dict[str, object] | None:
    """The pending human-in-the-loop request (action_requests), or ``None``.

    deepagents surfaces an approval pause as ``__interrupt__`` — a list of Interrupt
    objects whose ``.value`` is the HITL request. We only ever raise one such interrupt
    per turn, so the first one carries every gated tool call.
    """
    interrupts = result.get("__interrupt__")
    if not isinstance(interrupts, (list, tuple)) or not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else None
