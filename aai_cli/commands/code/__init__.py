from __future__ import annotations

from pathlib import Path

import typer

from aai_cli import command_registry, help_panels
from aai_cli.app.context import run_with_options
from aai_cli.code_agent.prompt import DEFAULT_MODEL
from aai_cli.commands.code import _exec as code_exec
from aai_cli.core import llm as gateway
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.CODE,
    order=10,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("code",),
)


@app.command(
    rich_help_panel=help_panels.CODE,
    epilog=examples_epilog(
        [
            ("Start a coding session in the current directory", "assembly code"),
            ("Kick off with an initial task", 'assembly code "add a --verbose flag"'),
            ("Run without approval prompts", 'assembly code --auto "fix the failing test"'),
            ("Point at another project", "assembly code --dir ../service"),
        ]
    ),
)
def code(
    ctx: typer.Context,
    prompt: str | None = typer.Argument(
        None, help="Initial task for the agent. Omit to just open the session"
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, "--model", help="LLM Gateway model", autocompletion=gateway.complete_model
    ),
    directory: Path = typer.Option(
        Path(),
        "--dir",
        "-C",
        help="Working directory the agent's file and shell tools operate in",
        file_okay=False,
        exists=True,
    ),
    auto: bool = typer.Option(
        False, "--auto", "-y", help="Skip approval prompts and run every tool automatically"
    ),
    docs: bool = typer.Option(
        True, "--docs/--no-docs", help="Connect to the AssemblyAI docs MCP server for reference"
    ),
    skills: bool = typer.Option(
        True, "--skills/--no-skills", help="Load installed agent skills (e.g. the assemblyai skill)"
    ),
    web: bool = typer.Option(
        True, "--web/--no-web", help="Enable Tavily web search when TAVILY_API_KEY is set"
    ),
    memory: bool = typer.Option(
        True, "--memory/--no-memory", help="Load and persist the agent's long-term memory"
    ),
    session: str = typer.Option(
        "default", "--session", help="Conversation session name (reuse to resume it)"
    ),
    persist: bool = typer.Option(
        True, "--persist/--fresh", help="Persist the session to disk (--fresh: ephemeral)"
    ),
    tui: bool = typer.Option(
        True, "--tui/--no-tui", help="Use the full-screen TUI (off: a plain read-eval loop)"
    ),
    voice: bool = typer.Option(
        True,
        "--voice/--no-voice",
        help="Speak to the agent and hear replies read back (readback needs the sandbox)",
    ),
) -> None:
    """Run a terminal coding agent backed by the AssemblyAI LLM Gateway

    An autonomous coding agent (built on the deepagents SDK) that reads, writes,
    and edits files, runs shell commands, searches the AssemblyAI docs, and can
    invoke the 'assembly' CLI itself — all in the working directory. It talks
    only to the AssemblyAI LLM Gateway. Mutating actions ask for approval unless
    you pass --auto.

    In an interactive terminal it defaults to voice: speak your request (mic ->
    streaming STT) and the agent's replies are read back aloud (sandbox only).
    Pass --no-voice for the keyboard TUI, or pipe input for the headless loop.
    """
    opts = code_exec.CodeOptions(
        prompt=prompt,
        model=model,
        root_dir=directory,
        auto=auto,
        docs=docs,
        skills=skills,
        web=web,
        memory=memory,
        session=session,
        persist=persist,
        tui=tui,
        voice=voice,
    )
    run_with_options(ctx, code_exec.run_code, opts, json=False)
