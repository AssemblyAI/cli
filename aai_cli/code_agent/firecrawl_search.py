"""Compatibility shim — firecrawl_search.py has moved to aai_cli.agent_cascade.firecrawl_search.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.firecrawl_search import (  # noqa: F401
    FIRECRAWL_API_KEY_ENV,
    WEB_SEARCH_TOOL_NAME,
    build_web_search_tool,
)
