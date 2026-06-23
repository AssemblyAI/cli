"""System-prompt construction for the live voice agent's deepagents brain.

Split out of ``brain.py`` to keep each module within the file-length gate. The prompt is
tailored to the tools actually bound, so the model is only ever told about capabilities it
has — advertising a missing tool made it announce an action ("I'll search…") it then couldn't
take, leaving the turn hanging with no answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from aai_cli.agent_cascade import datetime_tool, weather_tool, webpage_tool
from aai_cli.agent_cascade.firecrawl_search import WEB_SEARCH_TOOL_NAME

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Closes every guidance variant: the reply is spoken, so it must stay short and plain.
_SPOKEN_TAIL = (
    "Your reply is read aloud, so keep it short and spoken — no markdown, lists, code, or raw URLs."
)

# Advertised when --files is on, so the model knows it can touch the launch directory (and the
# spoken tail still keeps replies short). Writes pause for the user's y/n; reads are immediate.
_FILE_CAPABILITY = (
    "read, write, and search files in your working directory, and run code to solve problems "
    "and operate on this project"
)

# When the session has *no* tools wired (e.g. no web search and the docs host is
# unreachable), the model must answer from its own knowledge — and crucially must not
# promise an action it can't take. Without this, telling it "you can search the web" while
# no search tool is bound makes it narrate "I'll search for that…" and then stop, so the
# answer never comes (the tool it announced was never actually available to call).
_NO_TOOLS_GUIDANCE = (
    "You have no external tools available, so answer from your own knowledge. Never say "
    "you will search the web, look something up, or fetch a page — you can't do any of "
    "that, so don't promise it; if a question needs information you don't have, say so "
    f"briefly instead. {_SPOKEN_TAIL}"
)


def _join_clause(parts: list[str]) -> str:
    """Join capability phrases into a readable clause: ``a``, ``a and b``, ``a, b, and c``."""
    *initial, last = parts
    if not initial:
        return last
    # Oxford comma only once there are three-or-more items (two or more lead the last).
    joiner = ", and " if initial[1:] else " and "
    return f"{', '.join(initial)}{joiner}{last}"


def _tool_capabilities(tools: Sequence[BaseTool]) -> list[str]:
    """The spoken-capability phrases backed by present built-in tools.

    The live agent's built-in legs are the keyless Open-Meteo weather tool, the read-a-URL
    tool (web page or PDF), and the system-clock date/time tool (all always present) plus
    Firecrawl web search (only when ``FIRECRAWL_API_KEY`` is set) — so the prompt advertises
    each only when the agent can really do it. Advertising a missing tool made it announce
    an action ("I'll search…") it then couldn't take.
    """
    names = {tool.name for tool in tools}
    capabilities: list[str] = []
    if WEB_SEARCH_TOOL_NAME in names:
        capabilities.append("search the web for current or unfamiliar facts")
    if weather_tool.WEATHER_TOOL_NAME in names:
        capabilities.append("tell someone the current weather and short forecast for a place")
    if webpage_tool.READ_URL_TOOL_NAME in names:
        capabilities.append("read a web page or PDF you have the URL for")
    if datetime_tool.DATETIME_TOOL_NAME in names:
        capabilities.append("tell you the current date and time")
    return capabilities


def _extra_capability(extra_tools: Sequence[BaseTool]) -> str | None:
    """The spoken-capability phrase for user-configured MCP tools, listing them by name.

    The deepagents graph already shows the model each tool's schema, so this only has to
    name the tools so the guidance doesn't claim "no external tools" when MCP tools are
    bound — and so the model knows to reach for them.
    """
    names = sorted(tool.name for tool in extra_tools)
    if not names:
        return None
    return f"use your connected tools ({', '.join(names)})"


def build_system_prompt(
    persona: str,
    *,
    tools: Sequence[BaseTool],
    extra_tools: Sequence[BaseTool] = (),
    files: bool = False,
) -> str:
    """The live agent's system prompt: the user's persona plus tool guidance.

    The guidance is tailored to the bound tools so the model is only told about
    capabilities it actually has — advertising a missing tool (web search without a
    ``FIRECRAWL_API_KEY``) made the agent announce an action it then couldn't take, leaving
    the turn hanging with no answer. ``tools`` are the built-in legs (web search, URL
    fetch, AssemblyAI docs); ``extra_tools`` are user-configured MCP tools, advertised
    generically by name. ``files`` advertises the launch-directory read/write capability
    (the ``--files`` filesystem tools). With no capabilities at all the model answers from
    its own knowledge.
    """
    capabilities = _tool_capabilities(tools)
    extra = _extra_capability(extra_tools)
    if extra is not None:
        capabilities.append(extra)
    if files:
        capabilities.append(_FILE_CAPABILITY)
    if not capabilities:
        return f"{persona}\n\n{_NO_TOOLS_GUIDANCE}"
    guidance = (
        f"You can use tools to help answer: {_join_clause(capabilities)}. Reach for a "
        "tool when a question needs fresh or external information; answer directly and "
        "instantly when you already know. Only offer to do what these tools allow — don't "
        f"say you'll search the web or look something up unless it's listed here. {_SPOKEN_TAIL}"
    )
    return f"{persona}\n\n{guidance}"
