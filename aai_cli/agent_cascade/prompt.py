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

# The persona is user-supplied and can pull against the operational rules — a verbose or
# strongly in-character persona ("a pirate who loves long tales") fights the spoken-brevity and
# honesty guidance. State once that the rules below outrank the persona's *style*, so a chatty
# persona can't override the constraints that keep the spoken agent short and truthful.
_PERSONA_LATCH = (
    "Stay in character, but the rules below override the persona's style when they conflict."
)

# Advertised when --files is on, so the model knows it can touch the launch directory (and the
# spoken tail still keeps replies short). Writes pause for the user's y/n; reads are immediate.
_FILE_CAPABILITY = (
    "read, write, and search files in your working directory, run code to solve problems "
    "and operate on this project, and delegate a bigger job to a helper"
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
    "briefly instead. For example, say you don't have that handy rather than offering to "
    f"look it up and then going quiet. {_SPOKEN_TAIL}"
)

# Closes the guidance whenever tools are bound: a spoken agent that narrates a success it
# never achieved is worse than one that admits it couldn't, so it must report what the tools
# actually did rather than inventing the result it expected.
_HONESTY_GUIDANCE = (
    "Don't claim you've done something until the tool actually returns; if a tool fails or "
    "finds nothing, say so briefly instead of inventing an answer. If a search or lookup comes "
    "back empty or thin, try once more with different wording before giving up."
)

# Added when --files is on: writing files and running code change the user's project and can't
# be undone by speaking, so the model must confirm first and not narrate a change as done
# before it has actually landed.
_FILE_SAFETY_GUIDANCE = (
    "Writing files and running code change this project and can't be undone — confirm out "
    "loud before anything destructive or irreversible, and never say a change landed until it has. "
    "Read a file before overwriting it, and prefer merging your change into what's there over "
    "replacing the whole file unless asked."
)

# Introduces the launch directory's AGENTS.md/CLAUDE.md when one is present, so the model treats
# it as project background to ground its answers rather than as another instruction to recite.
_PROJECT_CONTEXT_INTRO = (
    "The following is background on the project in your working directory, taken from its "
    "AGENTS.md/CLAUDE.md. Use it to ground your answers, but keep your reply short and spoken."
)


def _append_project_context(prompt: str, project_context: str | None) -> str:
    """Append the launch directory's instruction files to the prompt as project background."""
    if not project_context:
        return prompt
    return f"{prompt}\n\n{_PROJECT_CONTEXT_INTRO}\n\n{project_context}"


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
    project_context: str | None = None,
) -> str:
    """The live agent's system prompt: the user's persona plus tool guidance.

    The guidance is tailored to the bound tools so the model is only told about
    capabilities it actually has — advertising a missing tool (web search without a
    ``FIRECRAWL_API_KEY``) made the agent announce an action it then couldn't take, leaving
    the turn hanging with no answer. ``tools`` are the built-in legs (web search, URL
    fetch, AssemblyAI docs); ``extra_tools`` are user-configured MCP tools, advertised
    generically by name. ``files`` advertises the launch-directory read/write capability
    (the ``--files`` filesystem tools). With no capabilities at all the model answers from
    its own knowledge. Whenever tools are bound the guidance also tells the model to report
    tool outcomes honestly (never narrate a success the tool didn't return), and the
    ``--files`` path adds a warning to confirm before irreversible writes or code execution.
    ``project_context`` (the launch directory's AGENTS.md/CLAUDE.md) is appended as project
    background when present, so the agent's answers are grounded in the project it's run from.
    """
    capabilities = _tool_capabilities(tools)
    extra = _extra_capability(extra_tools)
    if extra is not None:
        capabilities.append(extra)
    if files:
        capabilities.append(_FILE_CAPABILITY)
    if not capabilities:
        return _append_project_context(
            f"{persona}\n\n{_PERSONA_LATCH} {_NO_TOOLS_GUIDANCE}", project_context
        )
    guidance = (
        f"You can use tools to help answer: {_join_clause(capabilities)}. Reach for a "
        "tool when a question needs fresh or external information; answer directly and "
        "instantly when you already know. Only offer to do what these tools allow — don't "
        f"say you'll search the web or look something up unless it's listed here. {_HONESTY_GUIDANCE}"
    )
    if files:
        guidance = f"{guidance} {_FILE_SAFETY_GUIDANCE}"
    return _append_project_context(
        f"{persona}\n\n{_PERSONA_LATCH} {guidance} {_SPOKEN_TAIL}", project_context
    )
