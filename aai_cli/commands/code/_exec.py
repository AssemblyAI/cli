"""Run logic for `assembly code`: launch the aider coding agent on the LLM Gateway.

The command module (aai_cli/commands/code/__init__.py) only parses argv — it builds a
``CodeOptions`` and hands it to ``run_code`` via ``context.run_with_options`` (the
options/run split, see AGENTS.md), so tests drive the env wiring and subprocess
launch by constructing options directly instead of round-tripping argv.

aider is OpenAI-compatible (it talks to any endpoint via litellm), so pointing it at
the gateway is three things: ``OPENAI_API_BASE`` = the active env's gateway base,
``OPENAI_API_KEY`` = the resolved key, and an ``openai/<model>`` model name. aider
applies edits by parsing its own text edit-formats out of plain chat completions, so
it needs nothing from the gateway beyond what `assembly llm` already uses.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import typer

from aai_cli.app.context import AppState
from aai_cli.core import env, environments
from aai_cli.core.errors import missing_dependency
from aai_cli.ui import output

# The external coding agent we launch. Shelled out to (not embedded) so its heavy,
# explicitly-unstable Python API never enters our locked deps or our gates.
AIDER_BIN = "aider"
# The strongest coding model on the gateway's roster (core.llm.KNOWN_MODELS). Override
# with --model; the gateway is the source of truth for what's actually accepted.
DEFAULT_MODEL = "claude-opus-4-7"


@dataclass(frozen=True)
class CodeOptions:
    """Every `assembly code` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    model: str
    files: tuple[str, ...]


def _require_aider() -> None:
    """Fail with an install hint when the aider binary isn't on PATH.

    Checked before resolving the API key so a missing tool reports the install hint
    immediately instead of first dragging the user through a browser login.
    """
    if shutil.which(AIDER_BIN) is None:
        raise missing_dependency(
            "aider is required to launch a coding agent. "
            "Install it with `uv tool install aider-chat` (or `pipx install aider-chat`).",
            suggestion="See https://aider.chat for install options.",
        )


def _render_launching(data: dict[str, str]) -> str:
    """Human one-liner for the launch record (the JSON shape is emitted verbatim)."""
    return f"Launching aider via the AssemblyAI LLM Gateway (model {data['model']})."


def run_code(opts: CodeOptions, state: AppState, *, json_mode: bool) -> None:
    """Wire aider to the gateway and hand it the terminal in the current directory."""
    _require_aider()
    api_key = state.resolve_api_key()
    gateway = environments.active().llm_gateway_base
    output.emit(
        {"status": "launching", "tool": AIDER_BIN, "model": opts.model, "gateway": gateway},
        _render_launching,
        json_mode=json_mode,
    )
    child = env.child_env(OPENAI_API_BASE=gateway, OPENAI_API_KEY=api_key)
    argv = [AIDER_BIN, "--model", f"openai/{opts.model}", *opts.files]
    result = subprocess.run(argv, env=child, check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)
