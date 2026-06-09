# Guided Onboarding Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `aai onboard` — a guided first-run wizard that takes a newcomer from zero to a successful API request and tracks progress toward 100 requests.

**Architecture:** A new `aai_cli/onboard/` package holds a prompter abstraction (interactive vs non-interactive), ordered resumable section functions, a wizard orchestrator, and progress rendering. A per-profile request counter lives in `config.toml` and is incremented by the run commands. A new `aai_cli/commands/onboard.py` Typer sub-app wires it up; `main.py` registers it, reorders help, and offers the wizard on a credential-less bare `aai`.

**Tech Stack:** Python 3.12+, Typer, Rich, questionary, pydantic, pytest + syrupy. Run every tool through `uv run`.

---

## Conventions for every task

- Start each file with `from __future__ import annotations`.
- Errors → stderr via `output.error_console` / `CLIError`; data → stdout.
- Strict mypy on `aai_cli` (annotate everything). Tests may skip return annotations but must annotate fixture params (see [[mypy-warn-return-any-untyped-fixtures]] in repo memory — annotate locals fed by untyped fixtures).
- Run a single test: `uv run pytest tests/test_x.py::test_name -q`.
- After a feature lands, the full gate is `./scripts/check.sh` (must print `All checks passed.`).
- Never hand-edit `tests/__snapshots__/*.ambr`; regenerate with `uv run pytest --snapshot-update`.
- Commit after each task.

---

## File Structure

**Create:**
- `aai_cli/onboard/__init__.py` — package exports.
- `aai_cli/onboard/prompter.py` — `Prompter` protocol, `InteractivePrompter`, `NonInteractivePrompter`, `WizardCancelled`.
- `aai_cli/onboard/progress.py` — `GOAL`, `milestone_message`, `render_progress`.
- `aai_cli/onboard/sections.py` — `WizardContext`, `SectionResult`, the seven section functions.
- `aai_cli/onboard/wizard.py` — `run_onboarding`.
- `aai_cli/commands/onboard.py` — Typer sub-app (`onboard`, `onboard --status`).
- `tests/test_onboard_progress.py`, `tests/test_onboard_prompter.py`, `tests/test_onboard_sections.py`, `tests/test_onboard_wizard.py`, `tests/test_onboard_command.py`, `tests/test_onboard_counter.py`.

**Modify:**
- `aai_cli/config.py` — `Profile.requests_made` field + `get_requests_made` + `record_request`.
- `aai_cli/commands/transcribe.py`, `llm.py`, `stream.py`, `agent.py` — increment the counter on success.
- `aai_cli/commands/init.py` — extract `run_init(...)` so the wizard can scaffold without a launch.
- `aai_cli/commands/login.py` — point the post-login hint at `aai onboard`.
- `aai_cli/main.py` — register `onboard`, reorder `_COMMAND_ORDER`, bare-`aai` offer, update root epilog.
- `aai_cli/help_panels.py` — `QUICK_START` comment mentions `onboard`.
- `install.sh` — final hint becomes `aai onboard`.

---

## Task 1: Per-profile request counter in config

**Files:**
- Modify: `aai_cli/config.py` (add field at `Profile`, lines 25-35; add functions after `get_account_id`, ~line 230)
- Test: `tests/test_onboard_progress.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_progress.py
from __future__ import annotations

from pathlib import Path

import pytest

from aai_cli import config


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return tmp_path


def test_requests_made_starts_at_zero(tmp_config: Path) -> None:
    assert config.get_requests_made("default") == 0


def test_record_request_increments_and_persists(tmp_config: Path) -> None:
    assert config.record_request("default") == 1
    assert config.record_request("default") == 2
    # Survives a fresh read (new process would re-_load from disk).
    assert config.get_requests_made("default") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_progress.py -q`
Expected: FAIL with `AttributeError: module 'aai_cli.config' has no attribute 'get_requests_made'`.

- [ ] **Step 3: Add the field and functions**

In `aai_cli/config.py`, add to `Profile` (after `account_id: int | None = None`):

```python
    requests_made: int | None = None
```

After `get_account_id` (around line 230), add:

```python
def get_requests_made(profile: str) -> int:
    """How many billable API requests this profile has made through the CLI."""
    prof = _load().profiles.get(profile)
    return prof.requests_made or 0 if prof else 0


def record_request(profile: str) -> int:
    """Increment and persist this profile's CLI request count; return the new total.

    Powers the 'N of 100 requests' onboarding nudge. Counts only requests made
    through the CLI; `aai usage` is the authoritative account-wide figure.
    """
    _validate_profile(profile)
    cfg = _load()
    prof = cfg.profiles.setdefault(profile, Profile())
    prof.requests_made = (prof.requests_made or 0) + 1
    _dump(cfg)
    return prof.requests_made
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_progress.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/config.py tests/test_onboard_progress.py
git commit -m "feat(onboard): per-profile CLI request counter in config"
```

---

## Task 2: Progress rendering and milestone copy

**Files:**
- Create: `aai_cli/onboard/__init__.py`, `aai_cli/onboard/progress.py`
- Test: `tests/test_onboard_progress.py` (extend)

- [ ] **Step 1: Write the failing test (append to tests/test_onboard_progress.py)**

```python
from aai_cli.onboard import progress


def test_goal_is_100() -> None:
    assert progress.GOAL == 100


def test_milestone_message_fires_only_at_milestones() -> None:
    assert progress.milestone_message(1) is not None
    assert progress.milestone_message(10) is not None
    assert progress.milestone_message(50) is not None
    assert progress.milestone_message(100) is not None
    assert progress.milestone_message(2) is None
    assert progress.milestone_message(0) is None


def test_render_progress_mentions_count_goal_and_usage_pointer() -> None:
    rendered = progress.render_progress(7)
    assert "7" in rendered
    assert "100" in rendered
    assert "aai usage" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_progress.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.onboard'`.

- [ ] **Step 3: Create the package and module**

`aai_cli/onboard/__init__.py`:

```python
from __future__ import annotations
```

`aai_cli/onboard/progress.py`:

```python
from __future__ import annotations

from aai_cli import output

GOAL = 100

# Counts that earn a one-off cheer; keep keys in sync with the wizard's nudge.
_MILESTONES: dict[int, str] = {
    1: "You're activated 🎉 — your first request is in.",
    10: "10 requests in. You're getting the hang of it.",
    50: "Halfway to 100 — nice momentum.",
    GOAL: "100 requests — you're off the ground. 🚀",
}


def milestone_message(count: int) -> str | None:
    """Encouragement to show when a request count lands exactly on a milestone."""
    return _MILESTONES.get(count)


def render_progress(count: int) -> str:
    """A Rich-markup block: 'N of 100 API requests', any milestone, the usage pointer."""
    lines = [output.success(f"{count} of {GOAL} API requests")]
    cheer = milestone_message(count)
    if cheer:
        lines.append("  " + output.heading(cheer))
    lines.append("  " + output.hint("For your full account usage, run `aai usage`."))
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_progress.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/onboard/__init__.py aai_cli/onboard/progress.py tests/test_onboard_progress.py
git commit -m "feat(onboard): progress rendering and milestone copy"
```

---

## Task 3: Wire the counter into the run commands

**Files:**
- Modify: `aai_cli/commands/transcribe.py` (import line 23; body after line 423), `llm.py` (import line 10; after line 143), `stream.py` (import line 20; after line 389), `agent.py` (import line 19; after line 167)
- Test: `tests/test_onboard_counter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_counter.py
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli import client, config
from aai_cli.main import app


class _FakeTranscript:
    id = "t_123"
    status = "completed"
    text = "hello world"
    json_response = {"id": "t_123", "text": "hello world"}
    utterances = None


@pytest.fixture
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return tmp_path


def test_transcribe_increments_request_counter(
    tmp_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: _FakeTranscript())
    result = CliRunner().invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 0, result.output
    assert config.get_requests_made("default") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_counter.py -q`
Expected: FAIL with `assert 0 == 1`.

- [ ] **Step 3: Add the increment to each run command**

In `transcribe.py`, change the context import (line 23) to:

```python
from aai_cli.context import AppState, resolve_profile, run_command
```

Immediately after the `with output.status("Transcribing…", ...)` block (after line 423, before `if output_field is not None:`), add:

```python
        config.record_request(resolve_profile(state))
```

In `llm.py`, change the context import (line 10) to:

```python
from aai_cli.context import AppState, resolve_profile, run_command
```

In the one-shot `body`, after `content = gateway.content_of(response)` (line 143), add:

```python
        config.record_request(resolve_profile(state))
```

In `stream.py`, change the context import (line 20) to:

```python
from aai_cli.context import AppState, resolve_profile, run_command
```

After `_dispatch(session, opts)` (line 389), add:

```python
        config.record_request(resolve_profile(state))
```

In `agent.py`, change the context import (line 19) to:

```python
from aai_cli.context import AppState, resolve_profile, run_command
```

In the `try` block, on the line after `run_session(...)` (line 167), add:

```python
            config.record_request(resolve_profile(state))
```

(Note: a stream/agent session aborted with Ctrl-C before normal return is not counted — a known, acceptable MVP limitation. `--show-code` paths return earlier and are correctly not counted.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_counter.py -q`
Expected: PASS.

- [ ] **Step 5: Run the existing command suites to confirm no regressions**

Run: `uv run pytest tests/test_transcribe.py tests/test_stream_llm.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aai_cli/commands/transcribe.py aai_cli/commands/llm.py aai_cli/commands/stream.py aai_cli/commands/agent.py tests/test_onboard_counter.py
git commit -m "feat(onboard): count CLI API requests toward the 100 goal"
```

---

## Task 4: Prompter abstraction

**Files:**
- Create: `aai_cli/onboard/prompter.py`
- Test: `tests/test_onboard_prompter.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_prompter.py
from __future__ import annotations

import pytest

from aai_cli.errors import UsageError
from aai_cli.onboard.prompter import NonInteractivePrompter


def test_noninteractive_confirm_returns_default() -> None:
    p = NonInteractivePrompter()
    assert p.confirm("Run setup?", default=True) is True
    assert p.confirm("Run setup?", default=False) is False


def test_noninteractive_select_returns_default_or_first() -> None:
    p = NonInteractivePrompter()
    options = [("a", "Option A"), ("b", "Option B")]
    assert p.select("Pick", options) == "a"
    assert p.select("Pick", options, default="b") == "b"


def test_noninteractive_text_requires_default() -> None:
    p = NonInteractivePrompter()
    assert p.text("Name?", default="x") == "x"
    with pytest.raises(UsageError):
        p.text("Name?")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_prompter.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.onboard.prompter'`.

- [ ] **Step 3: Create the prompter module**

`aai_cli/onboard/prompter.py`:

```python
from __future__ import annotations

from typing import Protocol

import typer

from aai_cli import output
from aai_cli.errors import UsageError


class WizardCancelled(Exception):
    """Raised when the user aborts the wizard (Ctrl-C / empty selection)."""


class Prompter(Protocol):
    """How the wizard asks for input — one interface, interactive or not."""

    def section(self, title: str) -> None: ...
    def note(self, message: str) -> None: ...
    def confirm(self, title: str, *, default: bool = True) -> bool: ...
    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str: ...
    def text(self, title: str, *, default: str | None = None) -> str: ...


class InteractivePrompter:
    """Drives real terminal prompts (questionary for select, Typer for the rest)."""

    def section(self, title: str) -> None:
        output.console.print("\n" + output.heading(title))

    def note(self, message: str) -> None:
        output.console.print(output.hint(message))

    def confirm(self, title: str, *, default: bool = True) -> bool:
        return typer.confirm(title, default=default)

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        import questionary

        choice = questionary.select(
            title,
            choices=[questionary.Choice(title=label, value=value) for value, label in options],
            default=default,
        ).ask()
        if choice is None:  # Ctrl-C
            raise WizardCancelled
        return str(choice)

    def text(self, title: str, *, default: str | None = None) -> str:
        return typer.prompt(title, default=default)


class NonInteractivePrompter:
    """Never blocks for input: returns defaults, logs choices, refuses when no default.

    Keeps the CLI pipeline-safe — `--json`, a piped stdin, or an agent run can call
    the wizard without it hanging on a prompt no human will answer.
    """

    def section(self, title: str) -> None:
        output.error_console.print(output.heading(title))

    def note(self, message: str) -> None:
        output.error_console.print(output.hint(message))

    def confirm(self, title: str, *, default: bool = True) -> bool:
        output.error_console.print(output.hint(f"{title} → {default} (non-interactive)"))
        return default

    def select(
        self, title: str, options: list[tuple[str, str]], *, default: str | None = None
    ) -> str:
        chosen = default if default is not None else options[0][0]
        output.error_console.print(output.hint(f"{title} → {chosen} (non-interactive)"))
        return chosen

    def text(self, title: str, *, default: str | None = None) -> str:
        if default is None:
            raise UsageError(
                f"'{title}' needs a value, but this is a non-interactive session.",
                suggestion="Re-run `aai onboard` in an interactive terminal.",
            )
        return default
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_prompter.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/onboard/prompter.py tests/test_onboard_prompter.py
git commit -m "feat(onboard): interactive/non-interactive prompter abstraction"
```

---

## Task 5: Extract reusable `run_init` from the init command

This lets the wizard scaffold a template without launching a blocking server.

**Files:**
- Modify: `aai_cli/commands/init.py` (extract body into `run_init`, lines 203-243)
- Test: `tests/test_init.py` (existing suite must still pass)

- [ ] **Step 1: Add `run_init` and have the command delegate to it**

In `aai_cli/commands/init.py`, add a module-level function (above the `init` command, after `_launch`):

```python
def run_init(
    state: AppState,
    *,
    template: str | None,
    directory: str | None,
    no_install: bool,
    no_open: bool,
    force: bool,
    here: bool,
    port: int,
    json_mode: bool,
    launch: bool = True,
) -> Path:
    """Scaffold (and optionally install/launch) a template; return the target dir.

    `launch=False` is for callers like the onboarding wizard that must not block on a
    running dev server — it stops after install and leaves the run command as a hint.
    """
    if not json_mode:
        output.console.print(
            f"[aai.heading]AssemblyAI CLI[/aai.heading] [aai.muted]{__version__}[/aai.muted]"
        )
    chosen = _resolve_template(template)
    target = _resolve_target(directory, chosen, here=here, force=force)

    api_key = keys.resolve_optional_api_key(profile=state.profile)
    report = _scaffold_report(chosen, target, api_key)

    use_uv = runner.has_uv()
    install_rows, will_launch = _install_step(
        target, no_install=no_install, api_key=api_key, use_uv=use_uv
    )
    report.extend(install_rows)

    if not no_install and api_key is None:
        report.append(
            {
                "name": "launch",
                "status": "skipped",
                "detail": f"no API key; run `aai login`, then: cd {target} && uv run uvicorn api.index:app",
            }
        )

    output.emit(report, lambda d: steps.render_steps(d, heading="Setup"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    if launch and will_launch:
        _launch(target, port=port, use_uv=use_uv, no_open=no_open, json_mode=json_mode)
    elif not json_mode:
        output.console.print(
            output.hint(f"Run `cd {escape(str(target))} && uv run uvicorn api.index:app`.")
        )
    return target
```

Replace the `init` command's `body` (lines 203-243) with a delegating call:

```python
    def body(state: AppState, json_mode: bool) -> None:
        run_init(
            state,
            template=template,
            directory=directory,
            no_install=no_install,
            no_open=no_open,
            force=force,
            here=here,
            port=port,
            json_mode=json_mode,
        )
```

- [ ] **Step 2: Run the existing init suite to verify behavior is unchanged**

Run: `uv run pytest tests/test_init.py -q`
Expected: PASS (no behavior change for the `aai init` command).

- [ ] **Step 3: Commit**

```bash
git add aai_cli/commands/init.py
git commit -m "refactor(init): extract run_init for reuse by the onboarding wizard"
```

---

## Task 6: Wizard context, results, and the welcome/auth/first-request sections

**Files:**
- Create: `aai_cli/onboard/sections.py`
- Test: `tests/test_onboard_sections.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_sections.py
from __future__ import annotations

from pathlib import Path

import pytest

from aai_cli import client, config
from aai_cli.context import AppState
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import NonInteractivePrompter
from aai_cli.onboard.sections import SectionResult, WizardContext


class _FakeTranscript:
    id = "t_1"
    status = "completed"
    text = "hello"
    utterances = None


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WizardContext:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def test_auth_skips_when_key_already_present(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    assert sections.auth(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED


def test_first_request_transcribes_sample_and_counts(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: _FakeTranscript())
    assert sections.first_request(NonInteractivePrompter(), ctx) is SectionResult.DONE
    assert config.get_requests_made("default") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_sections.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.onboard.sections'`.

- [ ] **Step 3: Create the sections module (context, results, first three sections)**

`aai_cli/onboard/sections.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import assemblyai as aai

from aai_cli import client, config, environments, output, transcribe_render
from aai_cli.context import AppState, persist_browser_login
from aai_cli.errors import NotAuthenticated
from aai_cli.onboard import progress
from aai_cli.onboard.prompter import Prompter


class SectionResult(Enum):
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WizardContext:
    state: AppState
    profile: str
    json_mode: bool


def _has_key(profile: str) -> bool:
    try:
        config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return False
    return True


def welcome(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    count = config.get_requests_made(ctx.profile)
    if count:
        prompter.section("Welcome back to AssemblyAI")
        output.console.print(progress.render_progress(count))
        return SectionResult.DONE
    prompter.section("Welcome to AssemblyAI")
    prompter.note(
        "This wizard signs you in, runs your first transcription, and helps you build."
    )
    return SectionResult.DONE


def auth(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    if _has_key(ctx.profile):
        prompter.note("Already signed in.")
        return SectionResult.SKIPPED
    prompter.section("Sign in")
    method = prompter.select(
        "How do you want to sign in?",
        [("browser", "Sign in with your browser (recommended)"), ("key", "Paste an API key")],
        default="browser",
    )
    env = environments.active().name
    if method == "key":
        key = prompter.text("Paste your AssemblyAI API key")
        if not client.validate_key(key):
            output.console.print(output.fail("That key was rejected."))
            return SectionResult.FAILED
        config.set_api_key(ctx.profile, key)
        config.set_profile_env(ctx.profile, env)
        return SectionResult.DONE
    prompter.note(f"No account yet? Create one at {environments.active().signup_url}")
    persist_browser_login(ctx.profile, env)
    return SectionResult.DONE


def first_request(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Your first transcription")
    api_key = config.resolve_api_key(profile=ctx.profile)
    with output.status("Transcribing the sample clip…", json_mode=ctx.json_mode):
        transcript = client.transcribe(
            api_key, client.SAMPLE_AUDIO_URL, config=aai.TranscriptionConfig()
        )
    count = config.record_request(ctx.profile)
    transcribe_render.render_transcript_result(transcript, output.console)
    output.console.print(progress.render_progress(count))
    return SectionResult.DONE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_sections.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/onboard/sections.py tests/test_onboard_sections.py
git commit -m "feat(onboard): welcome, auth, and first-request sections"
```

---

## Task 7: Environment, build-path, Claude Code, and next-steps sections

**Files:**
- Modify: `aai_cli/onboard/sections.py`
- Test: `tests/test_onboard_sections.py` (extend)

- [ ] **Step 1: Write the failing test (append)**

```python
from aai_cli.commands import init as init_cmd


def test_environment_is_non_blocking(ctx: WizardContext) -> None:
    # Even if checks warn/fail, the section never blocks the wizard.
    assert sections.environment(NonInteractivePrompter(), ctx) is SectionResult.DONE


def test_build_path_skip_choice_does_nothing(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def _fake_run_init(*a: object, **k: object) -> Path:
        nonlocal called
        called = True
        return Path(".")

    monkeypatch.setattr(init_cmd, "run_init", _fake_run_init)
    # NonInteractivePrompter.select returns the default; build_path's default is "skip".
    assert sections.build_path(NonInteractivePrompter(), ctx) is SectionResult.SKIPPED
    assert called is False


def test_next_steps_renders_progress(ctx: WizardContext) -> None:
    assert sections.next_steps(NonInteractivePrompter(), ctx) is SectionResult.DONE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_sections.py -q`
Expected: FAIL with `AttributeError: module 'aai_cli.onboard.sections' has no attribute 'environment'`.

- [ ] **Step 3: Add the four sections**

Append to `aai_cli/onboard/sections.py`. First extend the imports at the top:

```python
from aai_cli.commands import doctor as doctor_cmd
from aai_cli.commands import init as init_cmd
from aai_cli.commands import setup as setup_cmd
```

Then add:

```python
_BUILD_CHOICES = [
    ("audio-transcription", "Transcribe audio files (web app)"),
    ("live-captions", "Live captions from streaming audio"),
    ("voice-agent", "A two-way voice agent"),
    ("skip", "Just the CLI for now"),
]


def environment(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Environment check")
    checks = [
        doctor_cmd._check_python(),
        doctor_cmd._check_ffmpeg(),
        doctor_cmd._check_audio(),
    ]
    output.console.print(doctor_cmd._render({"ok": True, "checks": checks}))
    prompter.note("Warnings here only affect live streaming and the voice agent.")
    return SectionResult.DONE


def build_path(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("What do you want to build?")
    choice = prompter.select("Pick a starting point", _BUILD_CHOICES, default="skip")
    if choice == "skip":
        return SectionResult.SKIPPED
    if not prompter.confirm(f"Scaffold the '{choice}' app now?", default=True):
        prompter.note(f"You can run `aai init {choice}` whenever you're ready.")
        return SectionResult.SKIPPED
    # launch=False: never block the wizard on a running dev server.
    init_cmd.run_init(
        ctx.state,
        template=choice,
        directory=None,
        no_install=False,
        no_open=True,
        force=False,
        here=False,
        port=3000,
        json_mode=ctx.json_mode,
        launch=False,
    )
    return SectionResult.DONE


def claude_code(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Coding agent (optional)")
    if not prompter.confirm("Wire up Claude Code (docs MCP + skills)?", default=False):
        return SectionResult.SKIPPED
    steps = [
        setup_cmd._install_mcp("user", False),
        setup_cmd._install_skill(False),
        setup_cmd._install_cli_skill(False),
    ]
    output.console.print(setup_cmd._render({"steps": steps}))
    return SectionResult.DONE


def next_steps(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("You're set up")
    output.console.print(progress.render_progress(config.get_requests_made(ctx.profile)))
    output.console.print(output.hint("Transcribe a file:  aai transcribe <file>"))
    output.console.print(output.hint("Stream live audio:  aai stream"))
    output.console.print(output.hint("Build an app:       aai init"))
    return SectionResult.DONE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_sections.py -q`
Expected: PASS (all section tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/onboard/sections.py tests/test_onboard_sections.py
git commit -m "feat(onboard): environment, build-path, Claude Code, next-steps sections"
```

---

## Task 8: Wizard orchestrator

**Files:**
- Create: `aai_cli/onboard/wizard.py`
- Test: `tests/test_onboard_wizard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_wizard.py
from __future__ import annotations

from pathlib import Path

import pytest

from aai_cli import config
from aai_cli.context import AppState
from aai_cli.onboard import sections, wizard
from aai_cli.onboard.prompter import NonInteractivePrompter
from aai_cli.onboard.sections import SectionResult, WizardContext


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WizardContext:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return WizardContext(state=AppState(), profile="default", json_mode=False)


def test_auth_failure_stops_the_wizard(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sections, "welcome", lambda p, c: SectionResult.DONE)
    monkeypatch.setattr(sections, "auth", lambda p, c: SectionResult.FAILED)
    ran_after = False

    def _first(p: object, c: object) -> SectionResult:
        nonlocal ran_after
        ran_after = True
        return SectionResult.DONE

    monkeypatch.setattr(sections, "first_request", _first)
    code = wizard.run_onboarding(NonInteractivePrompter(), ctx)
    assert code == 4  # NotAuthenticated exit code
    assert ran_after is False


def test_happy_path_runs_all_sections(
    ctx: WizardContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("welcome", "auth", "first_request", "environment", "build_path",
                 "claude_code", "next_steps"):
        monkeypatch.setattr(sections, name, lambda p, c: SectionResult.DONE)
    assert wizard.run_onboarding(NonInteractivePrompter(), ctx) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_wizard.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.onboard.wizard'`.

- [ ] **Step 3: Create the wizard module**

`aai_cli/onboard/wizard.py`:

```python
from __future__ import annotations

from aai_cli import output
from aai_cli.errors import NotAuthenticated
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import Prompter, WizardCancelled
from aai_cli.onboard.sections import SectionResult, WizardContext


def run_onboarding(prompter: Prompter, ctx: WizardContext) -> int:
    """Run the ordered sections; return a process exit code.

    Auth is the one hard stop (no key → later sections can't run). Cancellation
    (Ctrl-C / empty pick) exits cleanly. The terminal cursor is always restored.
    """
    try:
        sections.welcome(prompter, ctx)
        if sections.auth(prompter, ctx) is SectionResult.FAILED:
            output.error_console.print(
                output.fail("Could not sign in. Run `aai onboard` again to retry.")
            )
            return NotAuthenticated().exit_code
        sections.first_request(prompter, ctx)
        sections.environment(prompter, ctx)
        sections.build_path(prompter, ctx)
        sections.claude_code(prompter, ctx)
        sections.next_steps(prompter, ctx)
        return 0
    except WizardCancelled:
        output.error_console.print(output.hint("Setup cancelled. Run `aai onboard` to resume."))
        return 130
    finally:
        output.console.show_cursor(True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_wizard.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/onboard/wizard.py tests/test_onboard_wizard.py
git commit -m "feat(onboard): wizard orchestrator with auth hard-stop and clean cancel"
```

---

## Task 9: The `aai onboard` command

**Files:**
- Create: `aai_cli/commands/onboard.py`
- Test: `tests/test_onboard_command.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_onboard_command.py
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app


@pytest.fixture(autouse=True)
def tmp_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    return tmp_path


def test_status_shows_progress_without_running_wizard() -> None:
    config.record_request("default")
    config.record_request("default")
    result = CliRunner().invoke(app, ["onboard", "--status"])
    assert result.exit_code == 0, result.output
    assert "2 of 100" in result.output


def test_onboard_is_listed_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert "onboard" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_command.py -q`
Expected: FAIL (no `onboard` command registered yet).

- [ ] **Step 3: Create the command**

`aai_cli/commands/onboard.py`:

```python
from __future__ import annotations

import sys

import typer

from aai_cli import config, help_panels, output
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.onboard import progress, wizard
from aai_cli.onboard.prompter import InteractivePrompter, NonInteractivePrompter, Prompter
from aai_cli.onboard.sections import WizardContext

app = typer.Typer()


def _build_prompter() -> Prompter:
    """A real prompter only when both ends are a TTY; otherwise never block."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        return InteractivePrompter()
    return NonInteractivePrompter()


@app.command(
    rich_help_panel=help_panels.QUICK_START,
    epilog=examples_epilog(
        [
            ("Run the guided setup", "aai onboard"),
            ("Show your progress toward 100 requests", "aai onboard --status"),
        ]
    ),
)
def onboard(
    ctx: typer.Context,
    status: bool = typer.Option(False, "--status", help="Show request progress and exit."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Guided setup: sign in, run your first transcription, and start building."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        if status:
            count = config.get_requests_made(profile)
            output.emit(
                {"requests_made": count, "goal": progress.GOAL},
                lambda _d: progress.render_progress(count),
                json_mode=json_mode,
            )
            return
        wiz_ctx = WizardContext(state=state, profile=profile, json_mode=json_mode)
        code = wizard.run_onboarding(_build_prompter(), wiz_ctx)
        if code != 0:
            raise typer.Exit(code=code)

    # auto_login=False: the wizard owns the sign-in step itself.
    run_command(ctx, body, json=json_out, auto_login=False)
```

- [ ] **Step 4: Register the command in `main.py`**

In `aai_cli/main.py`, add `onboard` to the command imports (line 15-30 block):

```python
    onboard,
```

And register it (after `app.add_typer(init.app)`, line 166):

```python
app.add_typer(onboard.app)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_command.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add aai_cli/commands/onboard.py aai_cli/main.py tests/test_onboard_command.py
git commit -m "feat(onboard): aai onboard command with --status"
```

---

## Task 10: Help ordering, panel comment, and bare-`aai` first-run offer

**Files:**
- Modify: `aai_cli/main.py` (`_COMMAND_ORDER` line 39-64; root `epilog` line 101-109; `main` callback body after line 148), `aai_cli/help_panels.py` (line 15)
- Test: `tests/test_onboard_command.py` (extend)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_onboard_sorts_first_in_quick_start() -> None:
    result = CliRunner().invoke(app, ["--help"])
    # onboard should appear before init in the Quick Start panel.
    assert result.output.index("onboard") < result.output.index("init")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_onboard_command.py::test_onboard_sorts_first_in_quick_start -q`
Expected: FAIL (init currently precedes onboard, which falls to alpha order).

- [ ] **Step 3: Reorder and update help text**

In `aai_cli/main.py`, change the start of `_COMMAND_ORDER` (line 40-41) from:

```python
    # Quick Start — zero-to-running onboarding
    "init",
```

to:

```python
    # Quick Start — zero-to-running onboarding
    "onboard",
    "init",
```

Update the root `epilog` (lines 102-108) to lead with onboard:

```python
    epilog=examples_epilog(
        [
            ("Guided setup (start here)", "aai onboard"),
            ("Transcribe a file", "aai transcribe call.mp3"),
            ("Scaffold a starter app", "aai init"),
        ]
    )
```

In `aai_cli/help_panels.py` line 15, update the comment:

```python
QUICK_START = "Quick Start"  # zero-to-running onboarding: onboard, init
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_onboard_command.py -q`
Expected: PASS.

- [ ] **Step 5: Add the bare-`aai` first-run offer**

The root callback runs for `aai` with no subcommand (Typer then shows help because `no_args_is_help=True`). To offer the wizard first, intercept in the callback only when there's no subcommand, no credentials, and an interactive TTY.

In `aai_cli/main.py`, at the end of the `main` callback (after line 148, the `env_override_warning` block), add:

```python
    _maybe_offer_onboarding(ctx, state)
```

Add this helper above `main` (after `_version_callback`, line 99):

```python
def _maybe_offer_onboarding(ctx: typer.Context, state: AppState) -> None:
    """On a bare, credential-less, interactive `aai`, offer the guided wizard.

    Never hijacks `--help` or any subcommand; declining falls through to the normal
    help screen. Silent in non-interactive sessions so pipelines/agents are unaffected.
    """
    import sys

    if ctx.invoked_subcommand is not None:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    from aai_cli import config
    from aai_cli.errors import NotAuthenticated

    try:
        config.resolve_api_key(profile=state.profile)
    except NotAuthenticated:
        pass
    else:
        return  # already has a key; show normal help
    if typer.confirm("Welcome to AssemblyAI. Run guided setup now?", default=True):
        from aai_cli.commands.onboard import _build_prompter
        from aai_cli.onboard import wizard
        from aai_cli.onboard.sections import WizardContext

        wiz_ctx = WizardContext(
            state=state, profile=state.resolve_profile(), json_mode=False
        )
        raise typer.Exit(code=wizard.run_onboarding(_build_prompter(), wiz_ctx))
```

- [ ] **Step 6: Write a test for the offer (append to tests/test_onboard_command.py)**

```python
def test_bare_aai_with_key_does_not_offer_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    # Non-interactive (CliRunner) input → the offer is skipped; help is shown.
    result = CliRunner().invoke(app, [])
    assert "Usage" in result.output or "Commands" in result.output
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_onboard_command.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aai_cli/main.py aai_cli/help_panels.py tests/test_onboard_command.py
git commit -m "feat(onboard): list onboard first and offer it on a bare first run"
```

---

## Task 11: Point existing entry points at `aai onboard`

**Files:**
- Modify: `aai_cli/commands/login.py` (line 55), `install.sh` (line ~41)
- Test: `tests/test_login.py` (existing) / snapshot

- [ ] **Step 1: Update the post-login hint**

In `aai_cli/commands/login.py`, change line 55 from:

```python
                + output.hint("Run `aai transcribe <file>` to make your first transcript.")
```

to:

```python
                + output.hint("Run `aai onboard` to finish setup, or `aai transcribe <file>`.")
```

- [ ] **Step 2: Update install.sh**

In `install.sh`, change the final hint (line ~41) from:

```sh
echo "Installed. Next: run 'aai login', then 'aai transcribe --sample'."
```

to:

```sh
echo "Installed. Next: run 'aai onboard'."
```

- [ ] **Step 3: Run any login tests**

Run: `uv run pytest tests/test_login.py -q`
Expected: PASS (update assertions if a test pins the old hint text).

- [ ] **Step 4: Commit**

```bash
git add aai_cli/commands/login.py install.sh tests/test_login.py
git commit -m "docs(onboard): route post-install and post-login hints to aai onboard"
```

---

## Task 12: Regenerate snapshots and run the full gate

**Files:**
- Modify: `tests/__snapshots__/test_cli_output_snapshots.ambr` (regenerated, never hand-edited)

- [ ] **Step 1: Regenerate snapshots**

Run: `uv run pytest --snapshot-update -q`
Expected: snapshots updated for `aai --help` (new ordering, onboard panel entry) and any new onboard help captured by the snapshot suite.

- [ ] **Step 2: Review the snapshot diff**

Run: `git diff tests/__snapshots__/test_cli_output_snapshots.ambr`
Expected: only `onboard`-related additions and the Quick Start reordering; no unrelated churn. (See [[syrupy-ambr-vs-whitespace-hooks]] — never let a whitespace hook touch this file.)

- [ ] **Step 3: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` Address anything it flags — especially `vulture` (the `doctor`/`setup` helpers the wizard now imports are no longer "unused"; if vulture flags the `_`-prefixed cross-module use, add a targeted allow or call them via small public wrappers rather than a blanket ignore), `xenon` (keep `run_onboarding` and each section under grade B), and `diff-cover` (100% patch coverage — add tests for any uncovered branch).

- [ ] **Step 4: Commit**

```bash
git add tests/__snapshots__/test_cli_output_snapshots.ambr
git commit -m "test(onboard): regenerate CLI help snapshots for aai onboard"
```

---

## Self-Review

**Spec coverage:**
- Prompter abstraction (interactive/non-interactive) → Task 4. ✓
- Resumable sections that self-skip → auth/`_has_key` skip (Task 6), build-path skip (Task 7); welcome greets returning users (Task 6). ✓
- Terminal restore / clean cancel → `run_onboarding` try/finally + `WizardCancelled` (Task 8). ✓
- Seven ordered sections (welcome, auth, first-request, env, build-path, Claude Code, next-steps) → Tasks 6-7, ordered in Task 8. ✓
- First request fires inside the wizard + counts → Task 6 `first_request`. ✓
- Local request counter incremented by run commands → Tasks 1, 3. ✓
- `aai onboard` + `--status` → Task 9. ✓
- First-run autodetect (bare `aai` offer, never hijacks help) → Task 10. ✓
- Help ordering (`onboard` first in Quick Start) + install.sh/login hints → Tasks 10, 11. ✓
- Tests + snapshots + gate → every task + Task 12. ✓
- Out of scope (server-side usage, conversational mode, auth backend changes, new templates) → respected; not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step shows real code. ✓

**Type consistency:** `SectionResult`/`WizardContext` defined in Task 6 and used unchanged in 7-9; `Prompter`/`NonInteractivePrompter`/`InteractivePrompter`/`WizardCancelled` defined in Task 4 and reused in 6-10; `run_init` signature defined in Task 5 and called identically in Task 7; `config.record_request`/`get_requests_made` defined in Task 1 and used in 2,3,6,7,9. ✓

**Known follow-ups (not blockers):** Ctrl-C'd stream/agent sessions aren't counted (Task 3 note). Build-path scaffolds without launching to avoid blocking the wizard (Task 7).
```
