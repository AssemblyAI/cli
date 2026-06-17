"""Run logic for `assembly code`: the options/run split (see AGENTS.md).

The command module parses argv into a frozen ``CodeOptions`` and hands it here. This
assembles the gateway model; the agent's tools (the `assembly` CLI tool, the docs MCP,
web search, URL fetch, ask-user); the skills + long-term-memory middleware; a persistent
SQLite checkpointer; and the compiled deepagents graph, then drives it through either the
Textual TUI (a TTY) or a plain Rich read-eval loop (headless).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from aai_cli.app.context import AppState
from aai_cli.code_agent.agent import CompiledAgent, build_agent
from aai_cli.code_agent.ask_tool import AskBridge, build_ask_tool
from aai_cli.code_agent.cli_tool import build_cli_tool, run_assembly
from aai_cli.code_agent.docs_mcp import load_docs_tools
from aai_cli.code_agent.fetch_tool import build_fetch_tool
from aai_cli.code_agent.memory import build_memory_middleware
from aai_cli.code_agent.model import build_model
from aai_cli.code_agent.prompt import DEFAULT_MODEL
from aai_cli.code_agent.render import RichRenderer, make_approver
from aai_cli.code_agent.session import CodeSession, run_repl
from aai_cli.code_agent.skills import build_skills_middleware
from aai_cli.code_agent.store import build_checkpointer
from aai_cli.code_agent.web_search import TAVILY_API_KEY_ENV, build_web_search_tool
from aai_cli.core import env, errors, stdio
from aai_cli.ui import output

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class CodeOptions:
    """Every `assembly code` flag as plain data."""

    prompt: str | None
    model: str = DEFAULT_MODEL
    root_dir: Path = Path()
    auto: bool = False
    docs: bool = True
    skills: bool = True
    web: bool = True
    memory: bool = True
    session: str = "default"
    persist: bool = True
    tui: bool = True


def _assemble_tools(api_key: str, opts: CodeOptions, bridge: AskBridge) -> list[BaseTool]:
    """The agent's extra tools: the CLI tool, docs MCP, web search, URL fetch, ask-user."""
    tools: list[BaseTool] = [
        build_cli_tool(lambda args: run_assembly(args, api_key=api_key)),
        build_fetch_tool(),
        build_ask_tool(bridge),
    ]
    if opts.docs:
        tools.extend(load_docs_tools())
    if opts.web:
        search = build_web_search_tool()
        if search is not None:
            tools.append(search)
    return tools


def _assemble_middlewares(opts: CodeOptions) -> list[AgentMiddleware]:
    """Skills + long-term memory middleware, in load order."""
    middlewares: list[AgentMiddleware] = []
    if opts.skills:
        skills = build_skills_middleware()
        if skills is not None:
            middlewares.append(skills)
    if opts.memory:
        middlewares.append(build_memory_middleware())
    return middlewares


def _build_agent(api_key: str, opts: CodeOptions, bridge: AskBridge) -> CompiledAgent:
    """Wire the gateway model + tools + middlewares + checkpointer into the agent."""
    return build_agent(
        model=build_model(api_key, model=opts.model),
        root_dir=opts.root_dir.resolve(),
        tools=_assemble_tools(api_key, opts, bridge),
        middlewares=_assemble_middlewares(opts),
        checkpointer=build_checkpointer(persist=opts.persist),
        auto_approve=opts.auto,
    )


def _confirm(name: str, args: dict[str, object]) -> bool:
    """Headless approval: print the pending tool call and read a y/N from stdin."""
    rendered = ", ".join(f"{key}={value!r}" for key, value in args.items())
    output.error_console.print(output.warn(f"Run {name}({rendered})? [y/N] "))
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _ask_repl(question: str) -> str:
    """Headless ask-user: print the agent's question and read the answer from stdin."""
    output.console.print(output.heading(f"Agent asks: {question}"))
    try:
        return input("» ")
    except EOFError:
        return ""


def _read_line() -> str | None:
    """Read one prompt line; ``None`` on EOF (Ctrl-D) to end the loop."""
    try:
        return input("» ")
    except EOFError:
        return None


def _web_note(opts: CodeOptions) -> str | None:
    """The "web search disabled" notice when --web is on but no Tavily key is set."""
    if opts.web and not env.get(TAVILY_API_KEY_ENV):
        return (
            "TAVILY_API_KEY is not set, so web search is disabled. Get a key at https://tavily.com"
        )
    return None


def _run_tui(agent: CompiledAgent, opts: CodeOptions, bridge: AskBridge) -> None:
    from aai_cli.code_agent.tui import CodeAgentApp

    # mouse=False leaves terminal mouse reporting off, so native text selection (and
    # copy/paste) works in the transcript and prompt; the UI is fully keyboard-driven.
    CodeAgentApp(
        agent=agent,
        ask_bridge=bridge,
        auto_approve=opts.auto,
        initial=opts.prompt,
        thread_id=opts.session,
        cwd=opts.root_dir.resolve(),
        web_note=_web_note(opts),
    ).run(mouse=False)


def _print_repl_banner(opts: CodeOptions) -> None:
    from aai_cli.code_agent import banner

    for row in banner.wordmark():
        output.console.print(f"[{banner.BRAND_HEX}]{row}[/]", highlight=False)
    output.console.print(output.muted(banner.version()))
    output.console.print(output.muted(f"Thread: {opts.session}"))
    output.console.print(banner.READY_LINE, style=banner.BRAND_HEX, highlight=False)
    output.console.print(output.muted(banner.TIP_LINE))


def _run_repl(agent: CompiledAgent, opts: CodeOptions, bridge: AskBridge) -> None:
    _print_repl_banner(opts)
    bridge.handler = _ask_repl
    session = CodeSession(
        agent=agent,
        sink=RichRenderer(),
        approver=make_approver(_confirm),
        thread_id=opts.session,
        auto_approve=opts.auto,
    )
    run_repl(session, read_line=_read_line, initial=opts.prompt)


def run_code(opts: CodeOptions, state: AppState, *, json_mode: bool) -> None:
    """Start an `assembly code` coding session from already-parsed flags."""
    del json_mode  # the coding agent has no JSON output mode; it is a live session
    api_key = state.resolve_api_key()
    bridge = AskBridge()
    agent = _build_agent(api_key, opts, bridge)
    use_tui = opts.tui and stdio.stdout_is_tty() and stdio.stdin_is_tty()
    try:
        if use_tui:
            _run_tui(agent, opts, bridge)
        else:
            _run_repl(agent, opts, bridge)
    except KeyboardInterrupt:
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
