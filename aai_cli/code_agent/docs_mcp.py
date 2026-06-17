"""Load the AssemblyAI docs MCP server's tools for the agent.

`assembly setup` registers the same hosted docs server with Claude Code over HTTP;
here we connect to it directly through ``langchain-mcp-adapters`` and hand its tools
to deepagents, so the coding agent can search the AssemblyAI documentation while it
works. Connecting is best-effort: a sandbox that blocks the host, or an offline run,
degrades to "no docs tools" with a caller-visible warning rather than a hard failure.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# The hosted docs MCP server (HTTP transport) — the same endpoint `assembly setup`
# wires into Claude Code.
DOCS_MCP_URL = "https://mcp.assemblyai.com/docs"
DOCS_MCP_NAME = "assemblyai-docs"


async def _fetch(url: str) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({DOCS_MCP_NAME: {"transport": "streamable_http", "url": url}})
    return await client.get_tools()


def load_docs_tools(url: str = DOCS_MCP_URL) -> list[BaseTool]:
    """Connect to the docs MCP server and return its tools, or ``[]`` if unreachable.

    The adapter's ``get_tools`` is async; we drive it with ``asyncio.run`` since the
    command path is synchronous. Any connection/transport failure is swallowed and
    surfaced as an empty list so a blocked network never aborts a coding session.
    """
    try:
        return asyncio.run(_fetch(url))
    except Exception:
        return []
