"""A keyless read-a-URL tool for the `assembly live` voice agent.

Reads a web page or PDF the agent has a URL for and returns its readable text, so
the live agent can read an article the user names or a link surfaced by web search.
It reuses :func:`aai_cli.core.webpage.fetch_article` — the same trafilatura HTML
extraction and pypdf PDF text extraction that backs ``assembly speak --url`` — so no
API key is needed and every live session has this capability.

The only network seam is :data:`Reader` (a ``url -> Article`` callable), injected in
tests so the whole flow runs with no sockets — the same shape ``weather_tool`` uses.
Failures never raise out to the graph: ``read_url`` catches them and returns a short
spoken apology so a fetch outage can't sink a live turn.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aai_cli.core.errors import UsageError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from aai_cli.core.webpage import Article

# The registered tool name. ``brain.py`` detects availability and labels the live-UI
# affordance by this name, so a test pins it.
READ_URL_TOOL_NAME = "read_url"

# A reader GETs a URL and returns the extracted Article. Injected in tests (the only net seam).
Reader = Callable[[str], "Article"]

# Cap the returned text so a long article or multi-page PDF can't blow the model's context
# budget. The body is source for the model to summarize aloud, so the exact cap is a tuning
# knob — a +-1 shift is behaviorally equivalent, so no test can kill that mutant.
_MAX_CHARS = 16000  # pragma: no mutate


def _read(url: str) -> Article:
    """Fetch and extract ``url`` via core.webpage (imported lazily to stay off startup)."""
    from aai_cli.core.webpage import fetch_article

    return fetch_article(url)


def _format(article: Article) -> str:
    """Render the article as ``title + readable text``, truncated to ``_MAX_CHARS``."""
    body = article.text
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS] + "\n…[truncated]"
    if article.title:
        return f"{article.title}\n\n{body}"
    return body


def build_read_url_tool(read: Reader = _read) -> BaseTool:
    """Wrap the URL reader as the ``read_url`` tool (``read`` injectable for tests)."""
    from langchain_core.tools import tool

    @tool(READ_URL_TOOL_NAME)
    def read_url(url: str) -> str:
        """Read a web page or PDF by URL and return its text. Use to read an article,
        document, or page you have the URL for (e.g. from a web-search result)."""
        try:
            return _format(read(url))
        except UsageError:
            # Bad URL or no readable text (scanned/image-only PDF, paywalled/JS page).
            return "I couldn't find readable text on that page."
        except Exception:
            # Any fetch failure (APIError: DNS/timeout/non-2xx, or anything else) must not
            # bubble into brain's "couldn't complete the turn" path. Mirrors weather_tool.
            return "I couldn't read that page right now."

    return read_url
