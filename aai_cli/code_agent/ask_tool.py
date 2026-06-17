"""An `ask_user` tool so the agent can ask the user a question mid-task.

deepagents-code ships an AskUser middleware; base deepagents does not, so we add a
small tool. The actual prompting is injected through an :class:`AskBridge`: the Rich
REPL reads a line, the Textual TUI pops an input modal, and tests script the answer —
the tool itself just calls the bridge, so it stays framework-agnostic. It is *not*
approval-gated (it is itself the user interaction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

ASK_TOOL_NAME = "ask_user"


def _unanswered(_question: str) -> str:
    """Default handler before a front-end registers one: no human is attached."""
    return "No user is available to answer; proceed with your best judgment."


@dataclass
class AskBridge:
    """A late-bound seam for asking the user a question.

    The agent (and its tools) are built before the front-end exists, so the tool
    captures this bridge and the REPL/TUI sets :attr:`handler` once it's running.
    """

    handler: Callable[[str], str] = field(default=_unanswered)

    def ask(self, question: str) -> str:
        return self.handler(question)


def build_ask_tool(bridge: AskBridge) -> BaseTool:
    """Wrap an :class:`AskBridge` as the ``ask_user`` tool."""
    from langchain_core.tools import tool

    @tool(ASK_TOOL_NAME)
    def ask_user(question: str) -> str:
        """Ask the user a clarifying question and return their answer. Use when you
        genuinely need information only the user has before continuing."""
        return bridge.ask(question)

    return ask_user
