"""Expose the AssemblyAI CLI to the agent as a tool.

The agent gets an ``assembly`` tool that runs *this* CLI as a subprocess
(``python -m aai_cli …``), so a coding task can transcribe a file, run an LLM
transform, list transcripts, etc. without the model hand-rolling shell quoting.

Secrets never ride argv (the project-wide rule): the resolved API key is injected
into the child's environment, never appended to the argument list, so it can't leak
into ``ps`` or the model's own transcript of the command it ran.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from aai_cli.core import config, env

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# The tool name the model calls and the approval flow gates on.
CLI_TOOL_NAME = "assembly"

# Cap captured output so a chatty command can't blow the model's context window.
_MAX_OUTPUT_CHARS = 20000
# Backstop so a hung command (e.g. a stuck network call) can't wedge the session.
_DEFAULT_TIMEOUT = 600

# A runner takes the CLI argument list and returns the combined, formatted output.
CliRunner = Callable[[list[str]], str]


def _truncate(text: str) -> str:
    """Clip captured output to the context-window budget, marking that we did."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n…[output truncated]"


def _format_result(proc: subprocess.CompletedProcess[str]) -> str:
    """Render a finished CLI run as text the model can read: exit code + both streams."""
    parts = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        parts.append(f"stdout:\n{proc.stdout.rstrip()}")
    if proc.stderr:
        parts.append(f"stderr:\n{proc.stderr.rstrip()}")
    return _truncate("\n".join(parts))


def run_assembly(args: list[str], *, api_key: str, timeout: float = _DEFAULT_TIMEOUT) -> str:
    """Run ``assembly <args>`` as a subprocess and return its formatted output.

    Invoked as ``python -m aai_cli`` so it's the very CLI in use, independent of
    whatever ``assembly`` may (or may not) be on PATH. The key is passed through the
    environment, never argv.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "aai_cli", *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env.child_env(**{config.ENV_API_KEY: api_key}),
        timeout=timeout,
        check=False,
    )
    return _format_result(proc)


def build_cli_tool(runner: CliRunner) -> BaseTool:
    """Wrap a :data:`CliRunner` as the ``assembly`` LangChain tool the agent can call.

    The runner is injected so the orchestration is tested without spawning a real
    subprocess; the command layer passes :func:`run_assembly` bound to the session's key.
    """
    from langchain_core.tools import tool

    @tool(CLI_TOOL_NAME)
    def assembly(arguments: list[str]) -> str:
        """Run the AssemblyAI CLI. Pass CLI arguments as a list of strings, e.g.
        ["transcribe", "audio.mp3", "--json"]. Returns the command's exit code and
        output. Do not include an API key — it is provided via the environment."""
        return runner(arguments)

    return assembly
