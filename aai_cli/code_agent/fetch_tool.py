"""A URL-fetch tool for the coding agent (deepagents-code parity).

Distinct from web *search* (Firecrawl): this fetches a specific URL the agent already
knows and returns its text. It is approval-gated (see ``MUTATING_TOOLS``) because an
arbitrary fetch can reach internal/SSRF targets, so the user confirms each one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

FETCH_TOOL_NAME = "fetch_url"

# Keep fetched pages inside the model's context budget.
_MAX_CHARS = 20000
_TIMEOUT = 30.0

# A fetcher takes a URL and returns the response text (injected for hermetic tests).
Fetcher = Callable[[str], str]


def fetch_url(url: str, *, timeout: float = _TIMEOUT) -> str:
    """GET ``url`` and return its (truncated) text body."""
    import httpx

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    text = response.text
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS] + "\n…[truncated]"


def build_fetch_tool(fetcher: Fetcher = fetch_url) -> BaseTool:
    """Wrap a :data:`Fetcher` as the ``fetch_url`` tool (injectable for tests)."""
    from langchain_core.tools import tool

    @tool(FETCH_TOOL_NAME)
    def fetch_url_tool(url: str) -> str:
        """Fetch a URL over HTTP(S) and return its text content. Use for reading a
        specific page or API response you already have the URL for."""
        return fetcher(url)

    return fetch_url_tool
