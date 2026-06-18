"""Optional Firecrawl web search for the coding and live voice agents.

Firecrawl grounds the agent with live web search, enabled when a ``FIRECRAWL_API_KEY``
is present in the environment. Search is read-only, so it is *not* gated behind the
approval flow. With no key set we simply omit the tool (the agent still has its URL
fetch and the AssemblyAI docs MCP), rather than erroring.

Both ``assembly code`` (approval-gated, opt-out via ``--no-web``) and the live voice
agent share this single search tool via Firecrawl's official LangChain integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aai_cli.core import env

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Firecrawl's SDK reads this from the environment; we gate on its presence so we never
# hand the agent a search tool that will fail on first use for lack of a key.
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"

# The name ``FirecrawlSearch`` registers itself under. The prompt builder detects
# web-search availability by this name, so a test pins it against the tool.
WEB_SEARCH_TOOL_NAME = "firecrawl_search"


def build_web_search_tool() -> BaseTool | None:
    """The Firecrawl web-search tool, or ``None`` when no ``FIRECRAWL_API_KEY`` is set."""
    if not env.get(FIRECRAWL_API_KEY_ENV):
        return None

    from langchain_firecrawl import FirecrawlSearch

    return FirecrawlSearch()
