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

aider has no MCP/skills system, so the AssemblyAI coding-agent skills are injected the
aider-native way instead: their SKILL.md text is written into one read-only conventions
file passed via ``--read`` (the aai-cli skill ships in the wheel; the assemblyai skill is
included when `assembly setup` has installed it on disk).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli.app import coding_agent
from aai_cli.app.context import AppState
from aai_cli.core import env, environments, llm
from aai_cli.core.errors import missing_dependency
from aai_cli.ui import output, theme

# The external coding agent we launch. Shelled out to (not embedded) so its heavy,
# explicitly-unstable Python API never enters our locked deps or our gates.
AIDER_BIN = "aider"
# The strongest coding model on the gateway's roster (core.llm.KNOWN_MODELS). Override
# with --model; the gateway is the source of truth for what's actually accepted.
DEFAULT_MODEL = "claude-opus-4-7"
# The cheap model aider uses for side tasks (commit messages, summaries, repo-map). Point
# it at the gateway's cheap default so those never bill the expensive main model.
WEAK_MODEL = llm.DEFAULT_MODEL


@dataclass(frozen=True)
class CodeOptions:
    """Every `assembly code` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    model: str
    files: tuple[str, ...]
    message: str | None


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


def _gather_skill_docs() -> list[tuple[str, str]]:
    """``(skill name, SKILL.md text)`` for each AssemblyAI skill to feed aider as context.

    The aai-cli skill ships in the wheel, so it is always injected; the assemblyai skill
    is injected only when `assembly setup` has installed it on disk (it isn't bundled).
    Only each skill's SKILL.md is included — auxiliary reference files are left out.
    """
    docs: list[tuple[str, str]] = [("aai-cli", coding_agent.bundled_cli_skill_doc())]
    assemblyai = coding_agent.skill_dir() / "SKILL.md"
    if assemblyai.is_file():
        docs.append(("assemblyai", assemblyai.read_text(encoding="utf-8")))
    return docs


def _theme_args() -> list[str]:
    """aider color flags mapped to the CLI's Cobolt brand theme (see ui/theme.py).

    Only the colors with a clear brand value are set — your input and the assistant get
    the brand/secondary accents, tool output recedes (muted), errors use the brand red,
    and code blocks reuse the CLI's own syntax theme. Warnings/completion-menu/dark-mode
    keep aider's defaults rather than invent contrast pairs for an unknown background.
    """
    return [
        "--user-input-color",
        theme.BRAND,
        "--assistant-output-color",
        theme.ACCENT,
        "--tool-output-color",
        theme.MUTED,
        "--tool-error-color",
        theme.ERROR,
        "--code-theme",
        "ansi_dark",
    ]


def _write_conventions(docs: list[tuple[str, str]], dest_dir: Path) -> Path:
    """Write the gathered skills into one aider read-only conventions file."""
    parts = [
        "# AssemblyAI conventions (auto-injected by `assembly code`)",
        "",
        "Read-only reference for working with AssemblyAI in this project.",
    ]
    for name, text in docs:
        parts += ["", f"## {name}", "", text]
    path = dest_dir / "assemblyai-conventions.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


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
    child = env.child_env(
        OPENAI_API_BASE=gateway,
        OPENAI_API_KEY=api_key,
        # aider has its own analytics + update notifier; the CLI owns both, so silence
        # aider's (every aider flag mirrors to AIDER_<FLAG>). Set as env defaults so a
        # user who really wants them can still re-enable via their own aider config.
        AIDER_ANALYTICS="false",
        AIDER_CHECK_UPDATE="false",
    )
    # The conventions file only needs to outlive the aider process (which we block on),
    # so a temp dir keeps it out of the user's project and cleans up on exit.
    with tempfile.TemporaryDirectory() as tmp:
        conventions = _write_conventions(_gather_skill_docs(), Path(tmp))
        argv = [
            AIDER_BIN,
            "--model",
            f"openai/{opts.model}",
            # Route side tasks to the cheap model and silence the "unknown model" warning
            # litellm raises for our gateway model ids (we don't ship fabricated metadata).
            "--weak-model",
            f"openai/{WEAK_MODEL}",
            "--no-show-model-warnings",
            *_theme_args(),
            *opts.files,
            "--read",
            str(conventions),
        ]
        if opts.message is not None:
            # One-shot, non-interactive: run the instruction and exit (scripting/headless).
            argv += ["--message", opts.message]
        result = subprocess.run(argv, env=child, check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)
