"""The observe/act loop — the pure heart of `assembly control`.

Given a stream of spoken instructions and three injected seams — a
:data:`Responder` (one LLM turn), an :data:`Executor` (run one action on the
host), and a :class:`Renderer` (surface progress) — the engine runs the
computer-use loop and owns no I/O of its own. That keeps it exercisable with
fakes: no model, microphone, subprocess, or macOS required.

Per spoken utterance it appends a user message, then loops: ask the model,
execute any tool calls it returns (feeding each result back as a tool message),
and stop when the model replies with no further tool call (its spoken answer) or
the per-turn step budget is exhausted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aai_cli.control import actions
from aai_cli.control.actions import Action, InvalidAction

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# A chat message in OpenAI shape (role/content plus tool fields). The gateway is
# OpenAI-compatible, so messages are built as the SDK's param type via dict
# literals — that keeps `bridge.complete` type-clean with no cast. The type only
# matters to the checker (lazy alias + TYPE_CHECKING import), so there is no
# runtime dependency on the OpenAI SDK here.
type Message = ChatCompletionMessageParam


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model emitted: its id, the action name, and parsed arguments."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class Reply:
    """A single model turn: spoken content plus any tool calls to run first."""

    content: str
    tool_calls: tuple[ToolCall, ...]


# One LLM turn: given the running message list, return the model's reply.
type Responder = Callable[[list[Message]], Reply]
# Execute one action on the host and return the helper's JSON result.
type Executor = Callable[[Action], dict[str, object]]


class Renderer(Protocol):
    """How the engine surfaces progress (printing, a TUI, JSON events…)."""

    def on_user(self, text: str) -> None:
        """A finalized spoken instruction was heard."""

    def on_action(self, action: Action) -> None:
        """An action is about to run on the host."""

    def on_result(self, action: Action, result: dict[str, object]) -> None:
        """An action finished, with the helper's result."""

    def on_refused(self, action: Action, reason: str) -> None:
        """A UI-mutating action was refused (e.g. `--dry-run`)."""

    def on_invalid(self, reason: str) -> None:
        """The model called an unknown/under-specified tool."""

    def on_reply(self, text: str) -> None:
        """The model's spoken reply that ends a turn."""


# Shown (as the turn's spoken reply) when a turn hits its step budget without
# the model settling on an answer — so a runaway loop ends with feedback.
STEP_LIMIT_REPLY = "I took several steps without finishing; let me know how to continue."


def _assistant_message(reply: Reply) -> Message:
    """The assistant message to append for ``reply`` (OpenAI tool-call shape)."""
    if reply.tool_calls:
        return {
            "role": "assistant",
            "content": reply.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ],
        }
    return {"role": "assistant", "content": reply.content or None}


def _tool_message(call_id: str, result: dict[str, object]) -> Message:
    """The tool-result message to append for a finished (or rejected) tool call."""
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}


def _dispatch(
    call: ToolCall,
    *,
    execute: Executor,
    renderer: Renderer,
    allow_mutate: bool,
) -> dict[str, object]:
    """Validate, gate, and (if allowed) run one tool call; return the JSON result.

    A bad call or a `--dry-run`-refused mutating action returns an ``ok: False``
    result instead of executing — the model sees the failure and can adapt.
    """
    try:
        action = actions.validate(call.name, call.arguments)
    except InvalidAction as exc:
        renderer.on_invalid(str(exc))
        return {"ok": False, "error": str(exc)}
    if not allow_mutate and not action.is_observe():
        reason = "dry-run is on: refused to perform a UI-changing action"
        renderer.on_refused(action, reason)
        return {"ok": False, "error": reason}
    renderer.on_action(action)
    result = execute(action)
    renderer.on_result(action, result)
    return result


def run_turn(
    user_text: str,
    history: list[Message],
    *,
    respond: Responder,
    execute: Executor,
    renderer: Renderer,
    max_steps: int,
    allow_mutate: bool,
) -> list[Message]:
    """Drive one spoken instruction to completion; return the extended history.

    Loops model→tools→model up to ``max_steps`` times, ending when the model
    replies with no tool calls (its spoken answer) or the budget is hit.
    """
    renderer.on_user(user_text)
    messages: list[Message] = [*history, {"role": "user", "content": user_text}]
    for _ in range(max_steps):
        reply = respond(messages)
        messages.append(_assistant_message(reply))
        if not reply.tool_calls:
            renderer.on_reply(reply.content)
            return messages
        for call in reply.tool_calls:
            result = _dispatch(call, execute=execute, renderer=renderer, allow_mutate=allow_mutate)
            messages.append(_tool_message(call.id, result))
    renderer.on_reply(STEP_LIMIT_REPLY)
    return messages


def run_session(
    transcripts: Iterable[str],
    *,
    system: str,
    respond: Responder,
    execute: Executor,
    renderer: Renderer,
    max_steps: int,
    allow_mutate: bool,
) -> None:
    """Run the control loop over a stream of spoken instructions until it ends.

    History (including the system prompt) carries across turns, so a follow-up
    like "click it" resolves against what was just observed.
    """
    history: list[Message] = [{"role": "system", "content": system}]
    for user_text in transcripts:
        history = run_turn(
            user_text,
            history,
            respond=respond,
            execute=execute,
            renderer=renderer,
            max_steps=max_steps,
            allow_mutate=allow_mutate,
        )
