"""Run logic for `assembly code`: the options/run split (see AGENTS.md).

The command module parses argv into a frozen ``CodeOptions`` and hands it here. This
assembles the gateway model; the agent's tools (the `assembly` CLI tool, the docs MCP,
web search, URL fetch, ask-user); the skills + long-term-memory middleware; a persistent
SQLite checkpointer; and the compiled deepagents graph, then drives it through one of
three front-ends: a voice loop (the default in a TTY — speak your request, hear the
reply), the full-screen Textual TUI, or a plain Rich read-eval loop (headless).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.markup import escape

from aai_cli.app.context import AppState
from aai_cli.code_agent.agent import CompiledAgent, build_agent
from aai_cli.code_agent.ask_tool import AskBridge, build_ask_tool
from aai_cli.code_agent.cli_tool import build_cli_tool, run_assembly
from aai_cli.code_agent.docs_mcp import load_docs_tools
from aai_cli.code_agent.events import AssistantText, Event
from aai_cli.code_agent.fetch_tool import build_fetch_tool
from aai_cli.code_agent.memory import build_memory_middleware
from aai_cli.code_agent.model import build_model
from aai_cli.code_agent.prompt import DEFAULT_MODEL
from aai_cli.code_agent.render import RichRenderer, make_approver
from aai_cli.code_agent.session import CodeSession, EventSink, run_repl
from aai_cli.code_agent.skills import build_skills_middleware
from aai_cli.code_agent.store import build_checkpointer
from aai_cli.code_agent.voice import AUDIO_ERROR_TYPES, VoiceSession, build_voice_session
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
    voice: bool = True


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
    # escape() the tool name/args: they're echoed for approval but may contain "[" that
    # Rich would parse as markup (or raise on). The user still sees the full action.
    output.error_console.print(output.warn(f"Run {escape(name)}({escape(rendered)})? [y/N] "))
    try:
        answer = input().strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _ask_repl(question: str) -> str:
    """Headless ask-user: print the agent's question and read the answer from stdin."""
    output.console.print(output.heading(f"Agent asks: {escape(question)}"))
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


def _announce_voice(renderer: RichRenderer, voice: VoiceSession) -> None:
    """One-time voice-mode notice, naming whether replies are read back (sandbox) or not."""
    if voice.readback:
        renderer.notice(
            "Voice mode on: speak your request; replies are read back aloud. Ctrl-C to quit."
        )
    else:
        renderer.notice(
            "Voice mode on: speak your request. Readback needs the sandbox (streaming TTS), "
            "so replies show as text. Ctrl-C to quit."
        )


def _voice_sink(renderer: RichRenderer, voice: VoiceSession) -> EventSink:
    """Render every event, and read the assistant's natural-language text back aloud."""

    def sink(event: Event) -> None:
        renderer(event)
        if isinstance(event, AssistantText):
            voice.speak(event.text)

    return sink


def _voice_read_line(voice: VoiceSession, renderer: RichRenderer) -> Callable[[], str | None]:
    """A read-line that captures a spoken turn, degrading to typed input if no mic exists.

    The first time the microphone can't be opened (no device, sounddevice missing) it
    prints a one-line notice and switches to ``input()`` for the rest of the session, so a
    voice-default run on a mic-less box still works instead of erroring out.
    """
    state = {"typed": False}

    def read_line() -> str | None:
        if state["typed"]:
            return _read_line()
        renderer.notice("Listening… (speak now)")
        try:
            line = voice.listen()
        except errors.CLIError as exc:
            if exc.error_type not in AUDIO_ERROR_TYPES:
                raise
            renderer.notice(f"No microphone available ({exc.message}); switching to typed input.")
            state["typed"] = True
            return _read_line()
        if line:
            renderer.notice(f"Heard: {line}")
        return line

    return read_line


def _run_voice(agent: CompiledAgent, opts: CodeOptions, bridge: AskBridge, api_key: str) -> None:
    _print_repl_banner(opts)
    voice = build_voice_session(api_key)
    renderer = RichRenderer()
    _announce_voice(renderer, voice)
    bridge.handler = _ask_repl  # spoken clarifications still fall back to the keyboard
    session = CodeSession(
        agent=agent,
        sink=_voice_sink(renderer, voice),
        approver=make_approver(_confirm),
        thread_id=opts.session,
        auto_approve=opts.auto,
    )
    run_repl(session, read_line=_voice_read_line(voice, renderer), initial=opts.prompt)


def run_code(opts: CodeOptions, state: AppState, *, json_mode: bool) -> None:
    """Start an `assembly code` coding session from already-parsed flags."""
    del json_mode  # the coding agent has no JSON output mode; it is a live session
    api_key = state.resolve_api_key()
    bridge = AskBridge()
    agent = _build_agent(api_key, opts, bridge)
    interactive = stdio.stdout_is_tty() and stdio.stdin_is_tty()
    try:
        if opts.voice and interactive:
            _run_voice(agent, opts, bridge, api_key)
        elif opts.tui and interactive:
            _run_tui(agent, opts, bridge)
        else:
            _run_repl(agent, opts, bridge)
    except KeyboardInterrupt:
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
