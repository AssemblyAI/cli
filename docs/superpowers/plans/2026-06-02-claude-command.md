# `aai claude` Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `aai claude` command group that wires Claude Code up to AssemblyAI's docs MCP server and the `assemblyai-skill`, with `install`, `status`, and `remove` subcommands.

**Architecture:** A new Typer sub-app in `assemblyai_cli/commands/claude.py`, registered in `main.py`. It shells out to `claude mcp …` for the remote docs MCP server and to `npx skills add …` for the skill. Each artifact is handled by an independent step that detects its required tool with `shutil.which`, runs via `subprocess.run` (captured), and returns a `{"name", "status", "detail"}` dict. Output and error handling reuse the existing `context.run_command` / `output.emit` conventions.

**Tech Stack:** Python 3, Typer, Rich, pytest. Subprocess orchestration via stdlib `subprocess`/`shutil`/`pathlib`. No new runtime dependencies.

---

## Background for the implementer

Read these before starting:

- `assemblyai_cli/commands/samples.py` — the closest existing command. Shows the `typer.Typer()` sub-app pattern, `run_command(ctx, body, json=json_out)`, `output.emit(data, human_renderer, json_mode=...)`, and raising `CLIError` for failures.
- `assemblyai_cli/context.py` — `run_command` catches `CLIError`, emits it, and exits with `err.exit_code`. Anything else (including `typer.Exit`) propagates.
- `assemblyai_cli/output.py` — `resolve_json(explicit=...)` returns `True` when `--json` is passed **or** stdout is not a TTY. Under `pytest`'s `CliRunner`, stdout is never a TTY, so **command output is JSON by default in tests** even without `--json`. Tests parse `json.loads(result.output)`.
- `assemblyai_cli/errors.py` — `CLIError(message, *, error_type, exit_code)` and `UsageError` (exit code 2).
- `assemblyai_cli/main.py` — sub-apps are registered with `app.add_typer(mod.app, name="...")`.
- `tests/test_samples.py` — test style: drive the CLI through `runner.invoke(app, [...])` and assert on `exit_code` and `output`.

Key facts (already verified against Claude Code 2.1.161):

- MCP install: `claude mcp add --transport http --scope <scope> assemblyai-docs https://mcp.assemblyai.com/docs`. The `name` comes before the URL.
- MCP presence/removal: `claude mcp get assemblyai-docs` (exits non-zero when absent), `claude mcp remove assemblyai-docs --scope <scope>`.
- Skill install: `npx skills add AssemblyAI/assemblyai-skill` (re-runnable; updates/de-dupes on its own). It installs into `~/.claude/skills/assemblyai/` (contains `SKILL.md`).
- There is **no** `claude install-skill` command. Do not use one.

All tests **mock** `shutil.which` and `subprocess.run` — they never invoke real `claude` or `npx`.

---

## File Structure

- **Create** `assemblyai_cli/commands/claude.py` — the `claude` sub-app: constants, the `_run` subprocess helper, per-artifact step functions, and the `install` / `status` / `remove` commands.
- **Modify** `assemblyai_cli/main.py` — import `claude` and register it.
- **Create** `tests/test_claude.py` — all tests, mirroring `test_samples.py`.
- **Modify** `README.md` — add an "AI coding agents" section.

---

## Task 1: `aai claude install`

**Files:**
- Create: `assemblyai_cli/commands/claude.py`
- Modify: `assemblyai_cli/main.py:6` (import) and `assemblyai_cli/main.py:31-35` (registration)
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_claude.py`:

```python
import json
import subprocess

from typer.testing import CliRunner

from assemblyai_cli.main import app

runner = CliRunner()


class FakeRun:
    """Records subprocess calls and returns canned CompletedProcess results.

    `returncodes` maps a command prefix tuple (the first N argv tokens) to a
    return code; the longest matching prefix wins, default 0.
    """

    def __init__(self, returncodes=None):
        self.calls = []
        self.returncodes = returncodes or {}

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(cmd)
        rc = 0
        best = -1
        for prefix, code in self.returncodes.items():
            n = len(prefix)
            if tuple(cmd[:n]) == prefix and n > best:
                rc, best = code, n
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout="", stderr="boom")


def _all_tools_present(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: f"/usr/bin/{tool}",
    )


def test_install_happy_path_runs_both_steps(monkeypatch):
    _all_tools_present(monkeypatch)
    # MCP not yet present -> `mcp get` returns non-zero.
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses == {"mcp": "installed", "skill": "installed"}

    assert [
        "claude", "mcp", "add", "--transport", "http",
        "--scope", "user", "assemblyai-docs", "https://mcp.assemblyai.com/docs",
    ] in fake.calls
    assert ["npx", "skills", "add", "AssemblyAI/assemblyai-skill"] in fake.calls


def test_install_scope_passthrough(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--scope", "project"])
    assert result.exit_code == 0
    assert [
        "claude", "mcp", "add", "--transport", "http",
        "--scope", "project", "assemblyai-docs", "https://mcp.assemblyai.com/docs",
    ] in fake.calls


def test_install_invalid_scope_exits_2(monkeypatch):
    _all_tools_present(monkeypatch)
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run", FakeRun()
    )
    result = runner.invoke(app, ["claude", "install", "--scope", "bogus"])
    assert result.exit_code == 2


def test_install_idempotent_when_mcp_present(monkeypatch):
    _all_tools_present(monkeypatch)
    # `mcp get` returns 0 -> already registered.
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "already"
    # No `mcp add` should have run.
    assert not any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_force_removes_then_adds(monkeypatch):
    _all_tools_present(monkeypatch)
    fake = FakeRun({("claude", "mcp", "get"): 0})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install", "--force"])
    assert result.exit_code == 0
    assert ["claude", "mcp", "remove", "assemblyai-docs", "--scope", "user"] in fake.calls
    assert any(c[:3] == ["claude", "mcp", "add"] for c in fake.calls)


def test_install_skips_mcp_when_claude_missing(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    fake = FakeRun()
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0  # skip is not a failure
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "skipped"
    assert statuses["skill"] == "installed"
    assert not any(c[0] == "claude" for c in fake.calls)


def test_install_skips_skill_when_npx_missing(monkeypatch):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "npx" else f"/usr/bin/{tool}",
    )
    fake = FakeRun({("claude", "mcp", "get"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["skill"] == "skipped"
    assert statuses["mcp"] == "installed"
    assert not any(c[0] == "npx" for c in fake.calls)


def test_install_failure_exits_nonzero(monkeypatch):
    _all_tools_present(monkeypatch)
    # mcp not present, but `mcp add` fails.
    fake = FakeRun({("claude", "mcp", "get"): 1, ("claude", "mcp", "add"): 1})
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "install"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    statuses = {s["name"]: s["status"] for s in payload["steps"]}
    assert statuses["mcp"] == "failed"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_claude.py -v`
Expected: FAIL — collection/import errors and `aai claude` not being a known command (the sub-app does not exist yet).

- [ ] **Step 3: Create the `claude` module**

Create `assemblyai_cli/commands/claude.py`:

```python
from __future__ import annotations

import shutil
import subprocess

import typer
from rich.markup import escape

from assemblyai_cli import output
from assemblyai_cli.context import run_command
from assemblyai_cli.errors import UsageError

app = typer.Typer(help="Wire up Claude Code for AssemblyAI (docs MCP + skill).")

MCP_NAME = "assemblyai-docs"
MCP_URL = "https://mcp.assemblyai.com/docs"
SKILL_REPO = "AssemblyAI/assemblyai-skill"
_VALID_SCOPES = ("user", "project", "local")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _mcp_present() -> bool:
    return _run(["claude", "mcp", "get", MCP_NAME]).returncode == 0


def _install_mcp(scope: str, force: bool) -> dict:
    if shutil.which("claude") is None:
        return {
            "name": "mcp",
            "status": "skipped",
            "detail": (
                "Claude Code not found. Install it (https://claude.com/claude-code), "
                f"then run: claude mcp add --transport http --scope {scope} "
                f"{MCP_NAME} {MCP_URL}"
            ),
        }
    if _mcp_present():
        if not force:
            return {"name": "mcp", "status": "already", "detail": f"{MCP_NAME} already registered"}
        _run(["claude", "mcp", "remove", MCP_NAME, "--scope", scope])
    proc = _run(
        ["claude", "mcp", "add", "--transport", "http", "--scope", scope, MCP_NAME, MCP_URL]
    )
    if proc.returncode != 0:
        return {"name": "mcp", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    return {"name": "mcp", "status": "installed", "detail": f"{MCP_NAME} @ {scope} scope"}


def _install_skill() -> dict:
    if shutil.which("npx") is None:
        return {
            "name": "skill",
            "status": "skipped",
            "detail": (
                "Node.js/npx not found. Install Node.js, then run: "
                f"npx skills add {SKILL_REPO}"
            ),
        }
    proc = _run(["npx", "skills", "add", SKILL_REPO])
    if proc.returncode != 0:
        return {"name": "skill", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    return {"name": "skill", "status": "installed", "detail": SKILL_REPO}


def _render_steps(data: object) -> str:
    steps = data["steps"]  # type: ignore[index]
    lines = [f"  {s['name']}: {s['status']} — {escape(str(s['detail']))}" for s in steps]
    return "AssemblyAI coding-agent setup:\n" + "\n".join(lines)


@app.command()
def install(
    ctx: typer.Context,
    scope: str = typer.Option(
        "user", "--scope", help="Claude Code config scope: user, project, or local."
    ),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already present."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Install the AssemblyAI docs MCP server and skill into Claude Code."""

    def body(_state, json_mode: bool) -> None:
        if scope not in _VALID_SCOPES:
            raise UsageError(
                f"Invalid --scope '{scope}'. Choose one of: {', '.join(_VALID_SCOPES)}."
            )
        steps = [_install_mcp(scope, force), _install_skill()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Register the sub-app in `main.py`**

In `assemblyai_cli/main.py`, add `claude` to the import on line 6:

```python
from assemblyai_cli.commands import claude, login, samples, stream, transcribe, transcripts
```

And register it alongside the other sub-apps (after line 31's block):

```python
app.add_typer(claude.app, name="claude")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_claude.py -v`
Expected: PASS (all 8 tests).

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/commands/claude.py assemblyai_cli/main.py tests/test_claude.py
git commit -m "feat(claude): add 'aai claude install' for Claude Code MCP + skill"
```

---

## Task 2: `aai claude status`

**Files:**
- Modify: `assemblyai_cli/commands/claude.py` (add `skill_dir`, `_mcp_status`, `_skill_status`, `status` command)
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claude.py`:

```python
def test_status_reports_both_installed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    skill = tmp_path / ".claude" / "skills" / "assemblyai"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    # `mcp get` returns 0 -> present.
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 0}),
    )

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "installed", "skill": "installed"}


def test_status_reports_not_installed(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))  # no skill dir created
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.subprocess.run",
        FakeRun({("claude", "mcp", "get"): 1}),
    )

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "not_installed", "skill": "not_installed"}


def test_status_mcp_unknown_when_claude_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "assemblyai_cli.commands.claude.shutil.which",
        lambda tool: None if tool == "claude" else f"/usr/bin/{tool}",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", FakeRun())

    result = runner.invoke(app, ["claude", "status"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses["mcp"] == "unknown"
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_claude.py -k status -v`
Expected: FAIL — `claude status` is not a known command.

- [ ] **Step 3: Implement `status`**

In `assemblyai_cli/commands/claude.py`, add a `Path` import at the top of the import block:

```python
from pathlib import Path
```

Add the skill-dir helper and status helpers below `_install_skill`:

```python
def skill_dir() -> Path:
    return Path.home() / ".claude" / "skills" / "assemblyai"


def _mcp_status() -> dict:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "unknown", "detail": "Claude Code not found"}
    present = _mcp_present()
    return {"name": "mcp", "status": "installed" if present else "not_installed", "detail": MCP_NAME}


def _skill_status() -> dict:
    present = (skill_dir() / "SKILL.md").exists()
    return {
        "name": "skill",
        "status": "installed" if present else "not_installed",
        "detail": str(skill_dir()),
    }
```

Add the command (after `install`):

```python
@app.command()
def status(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show whether the AssemblyAI MCP server and skill are wired into Claude Code."""

    def body(_state, json_mode: bool) -> None:
        steps = [_mcp_status(), _skill_status()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_claude.py -v`
Expected: PASS (all tests, including the three new `status` tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/commands/claude.py tests/test_claude.py
git commit -m "feat(claude): add 'aai claude status'"
```

---

## Task 3: `aai claude remove`

**Files:**
- Modify: `assemblyai_cli/commands/claude.py` (add `_remove_mcp`, `_remove_skill`, `remove` command)
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_claude.py`:

```python
def test_remove_unwinds_both(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    skill = tmp_path / ".claude" / "skills" / "assemblyai"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# AssemblyAI")
    fake = FakeRun({("claude", "mcp", "get"): 0})  # present -> removable
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "remove"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "removed", "skill": "removed"}
    assert ["claude", "mcp", "remove", "assemblyai-docs", "--scope", "user"] in fake.calls
    assert not skill.exists()


def test_remove_when_absent_is_not_an_error(monkeypatch, tmp_path):
    _all_tools_present(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))  # no skill dir
    fake = FakeRun({("claude", "mcp", "get"): 1})  # absent
    monkeypatch.setattr("assemblyai_cli.commands.claude.subprocess.run", fake)

    result = runner.invoke(app, ["claude", "remove"])
    assert result.exit_code == 0
    statuses = {s["name"]: s["status"] for s in json.loads(result.output)["steps"]}
    assert statuses == {"mcp": "not_installed", "skill": "not_installed"}
    assert not any(c[:3] == ["claude", "mcp", "remove"] for c in fake.calls)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_claude.py -k remove -v`
Expected: FAIL — `claude remove` is not a known command.

- [ ] **Step 3: Implement `remove`**

In `assemblyai_cli/commands/claude.py`, add the removal helpers below `_skill_status`:

```python
def _remove_mcp(scope: str) -> dict:
    if shutil.which("claude") is None:
        return {"name": "mcp", "status": "skipped", "detail": "Claude Code not found"}
    if not _mcp_present():
        return {"name": "mcp", "status": "not_installed", "detail": MCP_NAME}
    proc = _run(["claude", "mcp", "remove", MCP_NAME, "--scope", scope])
    if proc.returncode != 0:
        return {"name": "mcp", "status": "failed", "detail": (proc.stderr or proc.stdout).strip()}
    return {"name": "mcp", "status": "removed", "detail": MCP_NAME}


def _remove_skill() -> dict:
    target = skill_dir()
    if not target.exists():
        return {"name": "skill", "status": "not_installed", "detail": str(target)}
    shutil.rmtree(target)
    return {"name": "skill", "status": "removed", "detail": str(target)}
```

Add the command (after `status`):

```python
@app.command()
def remove(
    ctx: typer.Context,
    scope: str = typer.Option(
        "user", "--scope", help="Claude Code config scope the MCP was added under."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Remove the AssemblyAI MCP server and skill from Claude Code."""

    def body(_state, json_mode: bool) -> None:
        if scope not in _VALID_SCOPES:
            raise UsageError(
                f"Invalid --scope '{scope}'. Choose one of: {', '.join(_VALID_SCOPES)}."
            )
        steps = [_remove_mcp(scope), _remove_skill()]
        output.emit({"steps": steps}, _render_steps, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_claude.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add assemblyai_cli/commands/claude.py tests/test_claude.py
git commit -m "feat(claude): add 'aai claude remove'"
```

---

## Task 4: Docs + full-surface smoke test

**Files:**
- Modify: `README.md`
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_claude.py`:

```python
def test_claude_help_lists_all_subcommands():
    result = runner.invoke(app, ["claude", "--help"])
    assert result.exit_code == 0
    assert "install" in result.output
    assert "status" in result.output
    assert "remove" in result.output
```

- [ ] **Step 2: Run it to verify it passes (regression guard)**

Run: `pytest tests/test_claude.py::test_claude_help_lists_all_subcommands -v`
Expected: PASS — all three commands now exist from Tasks 1–3. (This test guards against accidental de-registration; it does not need a red phase.)

- [ ] **Step 3: Add the README section**

In `README.md`, after the `## Streaming` section, add:

```markdown
## AI coding agents

Wire Claude Code up to AssemblyAI's live docs (MCP server) and the AssemblyAI
skill so your agent writes current, correct integration code:

    aai claude install            # installs the docs MCP server + skill (user scope)
    aai claude status             # show what's wired up
    aai claude remove             # unwind both

`install` shells out to `claude mcp add` for the docs MCP server and to
`npx skills add` for the skill. Pass `--scope project` to scope the MCP server to
the current project instead of the whole machine. A missing `claude` or `npx` is
reported and skipped (with the manual command to run), not treated as an error.
```

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`
Expected: PASS (entire suite, including the existing tests).

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_claude.py
git commit -m "docs(claude): document 'aai claude' commands; smoke-test subcommands"
```

---

## Self-Review

**Spec coverage:**
- Command surface (`install`/`status`/`remove`, `--scope`/`--force`/`--json`) → Tasks 1–3. ✓
- Shell out: `claude mcp add` (MCP), `npx skills add` (skill) → Task 1. ✓
- User-scope default + override → Task 1 (`scope="user"`, `_VALID_SCOPES`). ✓
- Dependency detection + skip-with-guidance + independence + exit-code rule → Task 1 (`_install_mcp`/`_install_skill` `shutil.which` guards; `failed` raises `Exit(1)`, `skipped` does not). ✓
- Idempotency + `--force` → Task 1 (`_mcp_present`, force remove-then-add; `npx skills add` re-runnable). ✓
- `status` (MCP via `claude mcp get`, skill via `SKILL.md`, `unknown` when `claude` missing) → Task 2. ✓
- `remove` (`claude mcp remove`; delete skill dir; absent = `not_installed`) → Task 3. ✓
- Error handling via `CLIError`/`UsageError`/`run_command`; JSON shape `{"steps":[{name,status,detail}]}` → Tasks 1–3. ✓
- Status vocabulary (`installed`/`already`/`skipped`/`failed`/`removed`/`not_installed`/`unknown`) → exercised across Tasks 1–3. ✓
- Docs section + upstream-docs discrepancy noted → Task 4 README + spec note. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `skill_dir()`, `_run`, `_mcp_present`, `_install_mcp(scope, force)`, `_install_skill()`, `_render_steps`, `_mcp_status`, `_skill_status`, `_remove_mcp(scope)`, `_remove_skill()`, and constants `MCP_NAME`/`MCP_URL`/`SKILL_REPO`/`_VALID_SCOPES` are referenced consistently across tasks. Step dict keys (`name`/`status`/`detail`) are uniform. ✓
