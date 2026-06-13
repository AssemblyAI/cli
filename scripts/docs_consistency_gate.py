from __future__ import annotations

import re
import sys
from pathlib import Path

import typer

from aai_cli.main import app

# Docs-stay-in-sync gate, in the spirit of curl's "every option is documented" presubmit
# and numpy's refguide-check: the reference doc and the code must not drift apart. Three
# checks, all static and fast:
#   1. Environment-variable parity — every AAI_*/ASSEMBLYAI_* var the code reads is either
#      documented in REFERENCE.md or explicitly listed as internal here, and every such
#      documented var is actually read (no stale rows).
#   2. Exit-code parity — every numeric exit code the code returns is in REFERENCE.md's
#      exit-code table.
#   3. Command-reference validity — every `assembly <cmd> [<subcmd>]` example in the docs
#      names a real command (catches a doc that outlives a rename).

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = REPO_ROOT / "REFERENCE.md"
DOC_SOURCES = (REPO_ROOT / "README.md", REFERENCE)
PACKAGE = REPO_ROOT / "aai_cli"

# Vars the code reads that are deliberately undocumented: telemetry plumbing overrides and
# the scaffold's product-config vars (written into a generated app's .env, not CLI behavior).
INTERNAL_VARS = {
    "AAI_TELEMETRY_CLIENT_TOKEN",
    "AAI_TELEMETRY_INTAKE_URL",
    "AAI_MACOS_AUDIO_DEBUG",
    "ASSEMBLYAI_BASE_URL",
    "ASSEMBLYAI_LLM_GATEWAY_URL",
    "ASSEMBLYAI_STREAMING_HOST",
    "ASSEMBLYAI_AGENTS_HOST",
}

_VAR_RE = re.compile(r"\b((?:AAI|ASSEMBLYAI)_[A-Z0-9_]+)\b")
_DOC_VAR_RE = re.compile(r"`((?:AAI|ASSEMBLYAI)_[A-Z0-9_]+)`")
_EXIT_DOC_RE = re.compile(r"\|\s*`(\d+)`\s*\|")
_EXIT_CODE_RE = re.compile(r"exit_code\s*[=:]\s*(\d+)|Exit\(code=(\d+)\)")
_CMD_RE = re.compile(r"\bassembly\s+([a-z][\w-]*)(?:\s+([a-z][\w-]*))?")


def _package_sources() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in PACKAGE.rglob("*.py")
        if "templates" not in p.parts and p.name != "_version.py"
    )


def _env_var_errors() -> list[str]:
    code = _package_sources()
    code_vars = set(_VAR_RE.findall(code))
    doc_vars = set(_DOC_VAR_RE.findall(REFERENCE.read_text(encoding="utf-8")))
    return [
        f"env var {var} is read in code but not documented in REFERENCE.md"
        for var in sorted(code_vars - doc_vars - INTERNAL_VARS)
    ] + [
        f"env var {var} is documented in REFERENCE.md but never read in code"
        for var in sorted(doc_vars - code_vars - INTERNAL_VARS)
    ]


def _exit_code_errors() -> list[str]:
    documented = {int(m) for m in _EXIT_DOC_RE.findall(REFERENCE.read_text(encoding="utf-8"))}
    errors: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if "templates" in path.parts:
            continue
        for groups in _EXIT_CODE_RE.findall(path.read_text(encoding="utf-8")):
            code = int(next(g for g in groups if g))
            if code not in documented:
                rel = path.relative_to(REPO_ROOT)
                errors.append(f"exit code {code} used in {rel} is not in REFERENCE.md's table")
    return errors


def _command_tree() -> tuple[set[str], dict[str, set[str]]]:
    root = typer.main.get_command(app)
    commands = getattr(root, "commands", {})
    groups = {
        name: set(getattr(obj, "commands", {}))
        for name, obj in commands.items()
        if hasattr(obj, "commands")
    }
    return set(commands), groups


def _command_ref_errors() -> list[str]:
    top, groups = _command_tree()
    errors: list[str] = []
    for doc in DOC_SOURCES:
        for cmd, sub in _CMD_RE.findall(doc.read_text(encoding="utf-8")):
            if cmd not in top:
                errors.append(f"{doc.name}: `assembly {cmd}` names an unknown command")
            elif sub and cmd in groups and sub not in groups[cmd]:
                errors.append(f"{doc.name}: `assembly {cmd} {sub}` names an unknown subcommand")
    return errors


def main() -> int:
    errors = _env_var_errors() + _exit_code_errors() + _command_ref_errors()
    if not errors:
        sys.stdout.write("Docs and code agree (env vars, exit codes, command references).\n")
        return 0
    for err in errors:
        sys.stdout.write(f"{err}\n")
    sys.stdout.write("Update REFERENCE.md/README.md (or the INTERNAL_VARS allowlist) to match.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
