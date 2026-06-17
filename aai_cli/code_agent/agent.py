"""Assemble the deepagents graph for `assembly code`.

Wires the gateway model to deepagents' built-in coding toolset (filesystem + shell,
rooted at the working directory via a `LocalShellBackend`), plus the custom `assembly`
CLI tool and any MCP/docs tools, the installed-skills middleware, and human-in-the-loop
approval on the mutating tools. The compiled graph is driven turn-by-turn from
`session.py`; an `InMemorySaver` checkpointer gives both conversation memory and the
interrupt/resume the approval flow needs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from aai_cli.code_agent.cli_tool import CLI_TOOL_NAME
from aai_cli.code_agent.fetch_tool import FETCH_TOOL_NAME
from aai_cli.code_agent.prompt import build_system_prompt

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver

# The tools whose effects reach outside the model — file writes, edits, arbitrary
# shell, the AssemblyAI CLI (which can spend account credits), and URL fetches (which
# can reach internal/SSRF targets). Each is gated behind human approval unless the
# session opts into --auto.
MUTATING_TOOLS = ("write_file", "edit_file", "execute", CLI_TOOL_NAME, FETCH_TOOL_NAME)


class CompiledAgent(Protocol):
    """The slice of the compiled langgraph graph the session drives.

    A structural type so we needn't name langgraph's deeply-generic
    ``CompiledStateGraph`` (and don't drag its type params through our code).
    """

    def invoke(
        self, input: object, config: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Run one step of the graph, returning the updated state (incl. messages)."""


def _interrupt_config(*, auto_approve: bool) -> dict[str, bool] | None:
    """The ``interrupt_on`` map: approve every mutating tool, or ``None`` under --auto."""
    if auto_approve:
        return None
    return dict.fromkeys(MUTATING_TOOLS, True)


def build_agent(
    *,
    model: BaseChatModel,
    root_dir: Path,
    tools: Sequence[BaseTool] = (),
    middlewares: Sequence[AgentMiddleware] = (),
    checkpointer: BaseCheckpointSaver | None = None,
    auto_approve: bool = False,
) -> CompiledAgent:
    """Compile the coding agent over ``root_dir`` with ``tools`` and ``middlewares``.

    ``model`` is the only network seam — tests pass a fake chat model so the real
    deepagents graph (filesystem + shell tools, approval, checkpointing) runs offline.
    ``checkpointer`` defaults to an in-memory saver (one ephemeral session); the command
    passes a SQLite saver for persistent, resumable sessions.
    """
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langgraph.checkpoint.memory import InMemorySaver

    # virtual_mode=True maps the model's "/"-rooted paths under root_dir and blocks
    # traversal escapes, so file ops and shell stay inside the working directory.
    backend = LocalShellBackend(root_dir=str(root_dir), virtual_mode=True)

    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=build_system_prompt(str(root_dir)),
        tools=list(tools),
        middleware=list(middlewares),
        interrupt_on=_interrupt_config(auto_approve=auto_approve),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
    )
