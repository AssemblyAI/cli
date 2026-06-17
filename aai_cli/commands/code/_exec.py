"""Run logic for `assembly code`: launch the opencode agent on the LLM Gateway.

The command module (aai_cli/commands/code/__init__.py) only parses argv — it builds a
``CodeOptions`` and hands it to ``run_code`` via ``context.run_with_options`` (the
options/run split, see AGENTS.md), so tests drive the config + subprocess launch by
constructing options directly instead of round-tripping argv.

opencode talks to any OpenAI-compatible endpoint, so we point it at the gateway by
generating an ``opencode.json`` that declares a custom ``@ai-sdk/openai-compatible``
provider (``options.baseURL`` = the active env's gateway, ``apiKey`` via ``{env:…}`` so
the key never lands in the file) and selects our model, then hand opencode that config
through the ``OPENCODE_CONFIG`` env var.

Unlike aider, opencode has native MCP + instructions, so we also (a) register the
AssemblyAI docs MCP and (b) feed the coding-agent skills via the config's ``instructions``
files. opencode is agentic (it tool-calls), so this path needs the gateway to proxy
tool/function-calling — see the launch caveats in the command help.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import typer

from aai_cli.app import coding_agent
from aai_cli.app.context import AppState
from aai_cli.core import env, environments
from aai_cli.core.errors import missing_dependency
from aai_cli.ui import output

# The external coding agent we launch. Shelled out to (not embedded): opencode is a
# TS/Go binary, so it can't be a Python dep anyway, and it lives in its own runtime.
OPENCODE_BIN = "opencode"
# The strongest coding model on the gateway's roster (core.llm.KNOWN_MODELS). Override
# with --model; the gateway is the source of truth for what's actually accepted.
DEFAULT_MODEL = "claude-opus-4-7"
# The provider id we give our generated provider; the model is referenced as
# "<PROVIDER_ID>/<model>" in the config's top-level `model` key.
PROVIDER_ID = "assemblyai"
# The AssemblyAI docs MCP (mirrors app/setup_exec.MCP_URL). opencode loads it natively, so
# the agent can query live docs — the integration the aider launcher couldn't offer.
DOCS_MCP_URL = "https://mcp.assemblyai.com/docs"
# The env var opencode reads the key from, referenced as {env:…} in the config so the key
# stays out of the on-disk config file (it rides only in the child environment).
KEY_ENV_VAR = "ASSEMBLYAI_API_KEY"


@dataclass(frozen=True)
class CodeOptions:
    """Every `assembly code` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    model: str
    message: str | None


def _require_opencode() -> None:
    """Fail with an install hint when the opencode binary isn't on PATH.

    Checked before resolving the API key so a missing tool reports the install hint
    immediately instead of first dragging the user through a browser login.
    """
    if shutil.which(OPENCODE_BIN) is None:
        raise missing_dependency(
            "opencode is required to launch a coding agent. "
            "Install it with `npm i -g opencode-ai` (or `brew install sst/tap/opencode`).",
            suggestion="See https://opencode.ai for install options.",
        )


def _render_launching(data: dict[str, str]) -> str:
    """Human one-liner for the launch record (the JSON shape is emitted verbatim)."""
    return f"Launching opencode via the AssemblyAI LLM Gateway (model {data['model']})."


def _gather_skill_docs() -> list[tuple[str, str]]:
    """``(skill name, SKILL.md text)`` for each AssemblyAI skill to feed opencode as context.

    The aai-cli skill ships in the wheel, so it is always injected; the assemblyai skill
    is injected only when `assembly setup` has installed it on disk (it isn't bundled).
    Only each skill's SKILL.md is included — auxiliary reference files are left out.
    """
    docs: list[tuple[str, str]] = [("aai-cli", coding_agent.bundled_cli_skill_doc())]
    assemblyai = coding_agent.skill_dir() / "SKILL.md"
    if assemblyai.is_file():
        docs.append(("assemblyai", assemblyai.read_text(encoding="utf-8")))
    return docs


def _write_conventions(docs: list[tuple[str, str]], dest_dir: Path) -> Path:
    """Write the gathered skills into one file opencode loads via `instructions`."""
    parts = [
        "# AssemblyAI conventions (auto-injected by `assembly code`)",
        "",
        "Read-only reference for working with AssemblyAI in this project.",
    ]
    for name, text in docs:
        parts += ["", f"## {name}", "", text]
    path = dest_dir / "assemblyai-skills.md"
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def _write_config(model: str, gateway: str, instructions: list[str], dest_dir: Path) -> Path:
    """Generate the opencode.json that wires the gateway provider, model, skills, and MCP."""
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            PROVIDER_ID: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "AssemblyAI LLM Gateway",
                # baseURL ends at /v1 (our llm_gateway_base does); apiKey via {env:…} keeps
                # the secret out of the file — opencode reads it from the child env.
                "options": {"baseURL": gateway, "apiKey": f"{{env:{KEY_ENV_VAR}}}"},
                "models": {model: {"name": model}},
            }
        },
        "model": f"{PROVIDER_ID}/{model}",
        "instructions": instructions,
        # Native MCP: the agent can query live AssemblyAI docs.
        "mcp": {"assemblyai-docs": {"type": "remote", "url": DOCS_MCP_URL, "enabled": True}},
        # The CLI owns updates; don't share code to opencode's service by default.
        "autoupdate": False,
        "share": "disabled",
    }
    path = dest_dir / "opencode.json"
    # indent is cosmetic — opencode parses any valid JSON, so the value can't be asserted.
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")  # pragma: no mutate
    return path


def run_code(opts: CodeOptions, state: AppState, *, json_mode: bool) -> None:
    """Wire opencode to the gateway via a generated config and hand it the terminal."""
    _require_opencode()
    api_key = state.resolve_api_key()
    gateway = environments.active().llm_gateway_base
    output.emit(
        {"status": "launching", "tool": OPENCODE_BIN, "model": opts.model, "gateway": gateway},
        _render_launching,
        json_mode=json_mode,
    )
    # The generated config + skills only need to outlive the opencode process (which we
    # block on), so a temp dir keeps them out of the user's project and cleans up on exit.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        conventions = _write_conventions(_gather_skill_docs(), tmp_dir)
        config = _write_config(opts.model, gateway, [str(conventions)], tmp_dir)
        child = env.child_env(**{KEY_ENV_VAR: api_key, "OPENCODE_CONFIG": str(config)})
        argv = [OPENCODE_BIN]
        if opts.message is not None:
            # One-shot, non-interactive: run the instruction and exit (scripting/headless).
            argv += ["run", opts.message]
        result = subprocess.run(argv, env=child, check=False)
    if result.returncode:
        raise typer.Exit(code=result.returncode)
