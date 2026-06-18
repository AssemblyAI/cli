"""Optional Tavily web search for the coding agent (matching deepagents-code).

dcode grounds the agent with Tavily web search; we offer the same as an opt-in tool,
enabled when a ``TAVILY_API_KEY`` is present in the environment. Search is read-only,
so it is *not* gated behind the approval flow. With no key set we simply omit the tool
(the agent still has the AssemblyAI docs MCP for reference), rather than erroring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aai_cli.core import env

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Tavily reads this from the environment; we gate on its presence so we never hand the
# agent a tool that will fail on first use for lack of a key.
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"

# The name ``TavilySearch`` registers itself under. Callers (e.g. the live agent's prompt
# builder) detect web-search availability by this name, so a test pins it against the tool.
WEB_SEARCH_TOOL_NAME = "tavily_search"

# A small result cap keeps search responses inside the model's context budget.
_DEFAULT_MAX_RESULTS = 5


def build_web_search_tool(max_results: int = _DEFAULT_MAX_RESULTS) -> BaseTool | None:
    """The Tavily web-search tool, or ``None`` when no ``TAVILY_API_KEY`` is set."""
    if not env.get(TAVILY_API_KEY_ENV):
        return None

    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=max_results)
