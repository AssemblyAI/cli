# Help Examples & Structured Error Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the `aai` CLI to parity with the Supabase CLI on two onboarding dimensions — embed copy-pasteable `EXAMPLES` in every command's `--help`, and give every actionable error a structured, separately-rendered `Suggestion:` line (in both human and JSON output).

**Architecture:** Two independent feature areas.
- **Examples:** A single helper (`aai_cli/help_text.py`) turns a list of `(description, command)` pairs into a Typer `epilog` string. Each command passes its examples via `epilog=`. The app already runs `rich_markup_mode="rich"`, which reflows single newlines, so the helper separates entries with blank lines and escapes markup. A reflection test walks the command tree and fails if any leaf command lacks an epilog — so future commands can't silently skip examples.
- **Errors:** The existing `CLIError` base + central `emit_error()` renderer already separate message from machine-type and route to stderr. We add one optional `suggestion` field that flows through `to_dict()` (JSON) and gets its own dimmed `Suggestion:` line (human). Centralized error helpers (`auth_failure()`, `NotAuthenticated`, `audio_missing_error()`) are upgraded once and auto-cover ~20 call sites; the remaining actionable sites split their embedded hints into `suggestion=`.

**Tech Stack:** Python 3.10+, Typer 0.25/0.26 (vendors its own click as `typer._click`), Rich 15, pytest. Tests live in `tests/`, run with `pytest`. Lint/type with `ruff` and `mypy`.

**Scoping decision (errors):** A `suggestion` is added only where there is a *distinct corrective action beyond restating the message*. Pure input-validation errors whose message already states the constraint are intentionally left unchanged: `aai_cli/config_builder.py` (e.g. "expects an integer, got X"), the mutually-exclusive-flag `UsageError`s in `stream.py`/`agent.py`/`llm.py` (e.g. "--device applies only to microphone input"), `auth/ams.py` raw server errors, `client.py` network/API-wrap errors (the auth cases there already route through `auth_failure()`), `commands/transcripts.py:34`, `commands/claude.py` scope errors, and `youtube.py` download failures (no fixed remedy). The mechanism is global; the *content* migration is scoped to actionable sites.

---

## Part A — Examples in `--help`

### Task A1: Examples helper

**Files:**
- Create: `aai_cli/help_text.py`
- Test: `tests/test_help_text.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_help_text.py
from aai_cli.help_text import examples_epilog


def test_examples_epilog_has_header_and_entries():
    epi = examples_epilog([("Do a thing", "aai do --thing")])
    assert "[bold]Examples[/bold]" in epi
    assert "[dim]Do a thing[/dim]" in epi
    assert "$ aai do --thing" in epi


def test_examples_epilog_blank_line_separates_entries():
    # rich_markup_mode="rich" reflows single newlines; blank lines keep each
    # entry on its own row.
    epi = examples_epilog([("First", "aai a"), ("Second", "aai b")])
    assert "\n\n" in epi
    assert epi.count("\n\n") >= 3  # header + 2 descs + 2 cmds, joined by blanks


def test_examples_epilog_escapes_markup_in_commands():
    # Brackets in example commands (jq filters, arrays) must not be parsed as
    # rich markup tags.
    epi = examples_epilog([("Filter JSON", "aai transcribe x -o json | jq '.utterances[]'")])
    assert "jq '.utterances\\[]'" in epi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_help_text.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'aai_cli.help_text'`

- [ ] **Step 3: Write minimal implementation**

```python
# aai_cli/help_text.py
from __future__ import annotations

from collections.abc import Sequence

from rich.markup import escape

# An (description, command) pair shown under a command's `--help`.
Example = tuple[str, str]


def examples_epilog(examples: Sequence[Example]) -> str:
    """Build a Typer ``epilog`` that renders each example on its own line.

    The app runs with ``rich_markup_mode="rich"``, which reflows single newlines
    into one paragraph but treats a blank line as a paragraph break. We join every
    line with a blank line so each renders on its own row, dim the descriptions so
    the commands stand out, and escape both so brackets in example commands (e.g.
    ``jq '.x[]'``) are not parsed as rich markup tags.
    """
    blocks = ["[bold]Examples[/bold]"]
    for description, command in examples:
        blocks.append(f"[dim]{escape(description)}[/dim]")
        blocks.append(f"$ {escape(command)}")
    return "\n\n".join(blocks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_help_text.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add aai_cli/help_text.py tests/test_help_text.py
git commit -m "feat(help): add examples_epilog helper for --help examples

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: Examples on `transcribe` and `stream`

**Files:**
- Modify: `aai_cli/commands/transcribe.py:33` (the `@app.command()` above `def transcribe`)
- Modify: `aai_cli/commands/stream.py:22` (the `@app.command()` above `def stream`)
- Test: `tests/test_transcribe.py`, `tests/test_stream_command.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transcribe.py`:

```python
def test_transcribe_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
    assert "aai transcribe" in result.output
```

Add to `tests/test_stream_command.py`:

```python
def test_stream_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["stream", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_transcribe.py::test_transcribe_help_has_examples tests/test_stream_command.py::test_stream_help_has_examples -v`
Expected: FAIL — `"Examples" in result.output` is False.

- [ ] **Step 3: Implement**

In `aai_cli/commands/transcribe.py`, add the import near the top (after the existing `from aai_cli...` imports):

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` (line 33, above `def transcribe`) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Transcribe a local file", "aai transcribe call.mp3"),
            ("Try it with the hosted sample", "aai transcribe --sample"),
            (
                "Diarize two speakers and redact PII",
                "aai transcribe call.mp3 --speaker-labels --speakers-expected 2 --redact-pii",
            ),
            ("Get just the text for a pipeline", "aai transcribe call.mp3 -o text"),
            ("Print equivalent Python instead of running", "aai transcribe call.mp3 --show-code"),
        ]
    )
)
```

In `aai_cli/commands/stream.py`, add the import near the top:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` (line 22, above `def stream`) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Stream from your microphone", "aai stream"),
            ("Stream the hosted sample", "aai stream --sample"),
            ("Summarize action items live as you talk", 'aai stream --llm "summarize action items"'),
            ("Print equivalent Python instead of running", "aai stream --show-code"),
        ]
    )
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_transcribe.py::test_transcribe_help_has_examples tests/test_stream_command.py::test_stream_help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/transcribe.py aai_cli/commands/stream.py tests/test_transcribe.py tests/test_stream_command.py
git commit -m "feat(help): add --help examples to transcribe and stream

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A3: Examples on `transcripts get` / `transcripts list`

**Files:**
- Modify: `aai_cli/commands/transcripts.py:15` (`@app.command()` above `def get`) and `:51` (`@app.command(name="list")` above `def list_`)
- Test: `tests/test_transcripts.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transcripts.py`:

```python
def test_transcripts_get_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["transcripts", "get", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output


def test_transcripts_list_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["transcripts", "list", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_transcripts.py -k help_has_examples -v`
Expected: FAIL

- [ ] **Step 3: Implement**

In `aai_cli/commands/transcripts.py`, add near the top imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` (line 15, above `def get`) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Fetch a transcript's text by id", "aai transcripts get 5551234-abcd"),
            ("Get the raw JSON", "aai transcripts get 5551234-abcd --json"),
        ]
    )
)
```

Replace `@app.command(name="list")` (line 51, above `def list_`) with:

```python
@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List your recent transcripts", "aai transcripts list"),
        ]
    ),
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_transcripts.py -k help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/transcripts.py tests/test_transcripts.py
git commit -m "feat(help): add --help examples to transcripts get/list

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A4: Examples on `agent` and `llm`

**Files:**
- Modify: `aai_cli/commands/agent.py:21` (`@app.command()` above `def agent`)
- Modify: `aai_cli/commands/llm.py:15` (`@app.command()` above `def llm`)
- Test: `tests/test_agent_command.py`, `tests/test_llm_command.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent_command.py`:

```python
def test_agent_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

Add to `tests/test_llm_command.py`:

```python
def test_llm_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["llm", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_agent_command.py::test_agent_help_has_examples tests/test_llm_command.py::test_llm_help_has_examples -v`
Expected: FAIL

- [ ] **Step 3: Implement**

In `aai_cli/commands/agent.py`, add near the top imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` (line 21, above `def agent`) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Start a live voice conversation", "aai agent"),
            ("Pick a voice and opening line", 'aai agent --voice james --greeting "Hi there"'),
            ("See available voices", "aai agent --list-voices"),
            ("Print equivalent Python instead of running", "aai agent --show-code"),
        ]
    )
)
```

In `aai_cli/commands/llm.py`, add near the top imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` (line 15, above `def llm`) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Summarize a past transcript", 'aai llm "summarize" --transcript-id 5551234-abcd'),
            ("Pipe any text in", 'echo "meeting notes" | aai llm "turn into action items"'),
            ("See available models", "aai llm --list-models"),
        ]
    )
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_agent_command.py::test_agent_help_has_examples tests/test_llm_command.py::test_llm_help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/agent.py aai_cli/commands/llm.py tests/test_agent_command.py tests/test_llm_command.py
git commit -m "feat(help): add --help examples to agent and llm

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A5: Examples on `login`, `logout`, `whoami`

**Files:**
- Modify: `aai_cli/commands/login.py:14` (`@app.command()` above `def login`), `:41` (above `def logout`), `:60` (above `def whoami`)
- Test: `tests/test_login.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_login.py`:

```python
import pytest


@pytest.mark.parametrize("cmd", ["login", "logout", "whoami"])
def test_auth_commands_help_has_examples(cmd):
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, [cmd, "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_login.py::test_auth_commands_help_has_examples -v`
Expected: FAIL (3 params)

- [ ] **Step 3: Implement**

In `aai_cli/commands/login.py`, add to the imports (after the existing `from aai_cli...` lines):

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` above `def login` (line 14) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Log in with your browser", "aai login"),
            ("Log in non-interactively (CI)", "aai login --api-key sk_..."),
        ]
    )
)
```

Replace `@app.command()` above `def logout` (line 41) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Clear stored credentials for the active profile", "aai logout"),
        ]
    )
)
```

Replace `@app.command()` above `def whoami` (line 60) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Show the active profile and whether its key works", "aai whoami"),
        ]
    )
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_login.py::test_auth_commands_help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/login.py tests/test_login.py
git commit -m "feat(help): add --help examples to login/logout/whoami

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A6: Examples on `doctor` and `samples list`/`create`

**Files:**
- Modify: `aai_cli/commands/doctor.py:190` (`@app.command()` above `def doctor`)
- Modify: `aai_cli/commands/samples.py:39` (`@app.command(name="list")`) and `:56` (`@app.command()` above `def create`)
- Test: `tests/test_doctor.py`, `tests/test_samples.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_doctor.py`:

```python
def test_doctor_help_has_examples():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

Add to `tests/test_samples.py`:

```python
import pytest


@pytest.mark.parametrize("argv", [["samples", "list"], ["samples", "create"]])
def test_samples_help_has_examples(argv):
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, [*argv, "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_doctor.py::test_doctor_help_has_examples tests/test_samples.py::test_samples_help_has_examples -v`
Expected: FAIL

- [ ] **Step 3: Implement**

In `aai_cli/commands/doctor.py`, add to imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` above `def doctor` (line 190) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Check your environment is ready", "aai doctor"),
        ]
    )
)
```

In `aai_cli/commands/samples.py`, add to imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command(name="list")` (line 39) with:

```python
@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List available starter scripts", "aai samples list"),
        ]
    ),
)
```

Replace `@app.command()` above `def create` (line 56) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Scaffold a transcribe starter script", "aai samples create transcribe"),
            ("Overwrite an existing script", "aai samples create transcribe --force"),
        ]
    )
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_doctor.py::test_doctor_help_has_examples tests/test_samples.py::test_samples_help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/doctor.py aai_cli/commands/samples.py tests/test_doctor.py tests/test_samples.py
git commit -m "feat(help): add --help examples to doctor and samples

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A7: Examples on `claude install`/`status`/`remove`

**Files:**
- Modify: `aai_cli/commands/claude.py:205` (`@app.command()` above `def install`), `:234` (above `def status`), `:248` (above `def remove`)
- Test: `tests/test_claude.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_claude.py`:

```python
import pytest


@pytest.mark.parametrize("sub", ["install", "status", "remove"])
def test_claude_help_has_examples(sub):
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["claude", sub, "--help"])
    assert result.exit_code == 0
    assert "Examples" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_claude.py::test_claude_help_has_examples -v`
Expected: FAIL (3 params)

- [ ] **Step 3: Implement**

In `aai_cli/commands/claude.py`, add to imports:

```python
from aai_cli.help_text import examples_epilog
```

Replace `@app.command()` above `def install` (line 205) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Wire AssemblyAI docs + skill into Claude Code", "aai claude install"),
            ("Install for the current project only", "aai claude install --scope project"),
        ]
    )
)
```

Replace `@app.command()` above `def status` (line 234) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Show whether Claude Code is wired up", "aai claude status"),
        ]
    )
)
```

Replace `@app.command()` above `def remove` (line 248) with:

```python
@app.command(
    epilog=examples_epilog(
        [
            ("Remove the AssemblyAI MCP server and skill", "aai claude remove"),
        ]
    )
)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_claude.py::test_claude_help_has_examples -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/claude.py tests/test_claude.py
git commit -m "feat(help): add --help examples to claude install/status/remove

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A8: Reflection guard — every leaf command has examples

**Files:**
- Test: `tests/test_help_examples_coverage.py` (create)

This capstone test walks the real command tree and fails if any leaf command (except `version`) is missing an epilog — so a future command can't ship without examples.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_help_examples_coverage.py
import typer

from aai_cli.main import app

# `version` is a trivial command with no flags; examples would be noise.
_EXEMPT = {"version"}


def _leaf_commands(click_cmd, prefix=()):
    """Yield (path_tuple, command) for every non-group command in the tree."""
    sub = getattr(click_cmd, "commands", None)
    if not sub:
        yield prefix, click_cmd
        return
    for name, child in sub.items():
        yield from _leaf_commands(child, (*prefix, name))


def test_every_leaf_command_has_examples_epilog():
    root = typer.main.get_command(app)
    missing = []
    for path, cmd in _leaf_commands(root):
        name = path[-1] if path else cmd.name
        if name in _EXEMPT:
            continue
        epilog = getattr(cmd, "epilog", None)
        if not (epilog and "Examples" in epilog):
            missing.append(" ".join(path))
    assert not missing, f"commands missing --help examples: {missing}"
```

- [ ] **Step 2: Run to verify it passes**

By this point Tasks A2–A7 have added epilogs to every leaf command. Run:

Run: `pytest tests/test_help_examples_coverage.py -v`
Expected: PASS. If it lists any command, add an `epilog=examples_epilog([...])` to that command following the Task A2 pattern, then re-run.

- [ ] **Step 3: Run the full help-examples test slice**

Run: `pytest tests/ -k "help_has_examples or examples_epilog or every_leaf_command" -v`
Expected: PASS (all)

- [ ] **Step 4: Lint & type-check the touched files**

Run: `ruff check aai_cli tests && mypy aai_cli`
Expected: no new errors.

- [ ] **Step 5: Commit**

```bash
git add tests/test_help_examples_coverage.py
git commit -m "test(help): guard that every leaf command ships --help examples

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Part B — Structured error suggestions

### Task B1: Add `suggestion` to the error model

**Files:**
- Modify: `aai_cli/errors.py:4-40` (the `CLIError`, `NotAuthenticated`, `APIError`, `UsageError` classes)
- Test: `tests/test_errors.py`

- [ ] **Step 1: Write/adjust the failing tests**

In `tests/test_errors.py`, replace `test_not_authenticated_defaults` (lines 4-8) and `test_to_dict_omits_none_transcript_id` (lines 20-22), and add new cases:

```python
def test_not_authenticated_defaults():
    err = NotAuthenticated()
    assert err.exit_code == 2
    assert err.error_type == "not_authenticated"
    assert err.message == "Not authenticated."
    assert err.suggestion == "Run 'aai login'."


def test_to_dict_includes_suggestion_when_present():
    err = CLIError("nope", error_type="generic", exit_code=1, suggestion="do this")
    assert err.to_dict() == {
        "error": {"type": "generic", "message": "nope", "suggestion": "do this"}
    }


def test_to_dict_omits_none_transcript_id_and_suggestion():
    err = CLIError("nope", error_type="generic", exit_code=1)
    assert err.to_dict() == {"error": {"type": "generic", "message": "nope"}}


def test_api_error_carries_suggestion():
    err = APIError("boom", suggestion="retry")
    assert err.to_dict() == {
        "error": {"type": "api_error", "message": "boom", "suggestion": "retry"}
    }
```

Add `UsageError` to the import line at the top of `tests/test_errors.py`:

```python
from aai_cli.errors import APIError, CLIError, NotAuthenticated, UsageError, is_auth_failure
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL — `CLIError.__init__` has no `suggestion` kwarg; `err.message` still includes "Run 'aai login'".

- [ ] **Step 3: Implement**

Replace `aai_cli/errors.py` lines 4-40 (the four classes) with:

```python
class CLIError(Exception):
    """Base error carrying an exit code, a machine-readable type, and an optional
    human suggestion for how to fix it."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "error",
        exit_code: int = 1,
        transcript_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.exit_code = exit_code
        self.transcript_id = transcript_id
        self.suggestion = suggestion

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {"type": self.error_type, "message": self.message}
        if self.suggestion is not None:
            body["suggestion"] = self.suggestion
        if self.transcript_id is not None:
            body["transcript_id"] = self.transcript_id
        return {"error": body}


class NotAuthenticated(CLIError):
    def __init__(
        self,
        message: str = "Not authenticated.",
        *,
        suggestion: str | None = "Run 'aai login'.",
    ) -> None:
        super().__init__(
            message, error_type="not_authenticated", exit_code=2, suggestion=suggestion
        )


class APIError(CLIError):
    def __init__(
        self,
        message: str,
        *,
        transcript_id: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(
            message,
            error_type="api_error",
            exit_code=1,
            transcript_id=transcript_id,
            suggestion=suggestion,
        )


class UsageError(CLIError):
    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        super().__init__(message, error_type="usage_error", exit_code=2, suggestion=suggestion)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/errors.py tests/test_errors.py
git commit -m "feat(errors): add optional suggestion field to CLIError model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B2: Render the `Suggestion:` line in human output

**Files:**
- Modify: `aai_cli/output.py:71-76` (`emit_error`)
- Test: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_output.py`:

```python
def test_emit_error_renders_suggestion_line(capsys):
    import types

    err = types.SimpleNamespace(
        message="bad thing",
        suggestion="try this instead",
        to_dict=lambda: {"error": {}},
    )
    output.emit_error(err, json_mode=False)
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "bad thing" in captured.err
    assert "Suggestion:" in captured.err
    assert "try this instead" in captured.err
    assert captured.out == ""


def test_emit_error_no_suggestion_line_when_absent(capsys):
    import types

    err = types.SimpleNamespace(message="bad thing", suggestion=None, to_dict=lambda: {"error": {}})
    output.emit_error(err, json_mode=False)
    captured = capsys.readouterr()
    assert "Suggestion:" not in captured.err
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_output.py -k suggestion -v`
Expected: FAIL — no "Suggestion:" line emitted.

- [ ] **Step 3: Implement**

Replace `aai_cli/output.py` lines 71-76 (`emit_error`) with:

```python
def emit_error(err: CLIError, *, json_mode: bool) -> None:
    # Always to stderr, so stdout stays clean for `aai … | next-tool` pipelines.
    if json_mode:
        print(json.dumps(err.to_dict(), default=str), file=sys.stderr)
    else:
        error_console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")
        suggestion = getattr(err, "suggestion", None)
        if suggestion:
            error_console.print(f"[aai.muted]Suggestion:[/aai.muted] {escape(suggestion)}")
```

(The `aai.muted` style already exists in the theme — `commands/doctor.py` uses `[aai.muted]fix:[/aai.muted]`.)

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_output.py -k suggestion -v`
Expected: PASS

- [ ] **Step 5: Verify JSON output already carries the suggestion**

The JSON branch calls `err.to_dict()`, which Task B1 already taught to include `suggestion`. Confirm with the existing error model test plus:

Run: `pytest tests/test_output.py tests/test_errors.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add aai_cli/output.py tests/test_output.py
git commit -m "feat(errors): render Suggestion line in human error output

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B3: Upgrade centralized error helpers (auto-covers ~20 sites)

These three helpers are reused across the codebase; splitting them once fixes every call site (all `auth_failure()` calls in `client.py`/`llm.py`/`agent/session.py`, every default `NotAuthenticated()` in `config.py`/`commands/login.py`/`commands/whoami`, and both audio import paths).

**Files:**
- Modify: `aai_cli/errors.py:57-70` (`REJECTED_KEY_MESSAGE`, `auth_failure`)
- Modify: `aai_cli/microphone.py:20-26` (`audio_missing_error`)
- Test: `tests/test_errors.py`, `tests/test_microphone.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_errors.py`:

```python
def test_auth_failure_splits_message_and_suggestion():
    from aai_cli.errors import auth_failure

    err = auth_failure()
    assert err.error_type == "not_authenticated"
    assert err.message == "Your API key was rejected."
    assert "aai login" in (err.suggestion or "")
    assert "ASSEMBLYAI_API_KEY" in (err.suggestion or "")
```

Add to `tests/test_microphone.py` (mirror its existing import style):

```python
def test_audio_missing_error_has_reinstall_suggestion():
    from aai_cli.microphone import audio_missing_error

    err = audio_missing_error()
    assert "sounddevice" in err.message
    assert err.suggestion is not None
    assert "pip install" in err.suggestion
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_errors.py::test_auth_failure_splits_message_and_suggestion tests/test_microphone.py::test_audio_missing_error_has_reinstall_suggestion -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Replace `aai_cli/errors.py` lines 57-70 (from `REJECTED_KEY_MESSAGE = (` through the end of `auth_failure`) with:

```python
REJECTED_KEY_MESSAGE = "Your API key was rejected."
REJECTED_KEY_SUGGESTION = "Run 'aai login' with a valid key, or set ASSEMBLYAI_API_KEY."


def is_auth_failure(exc: object) -> bool:
    """Heuristic: does this exception/error indicate rejected/invalid credentials?"""
    text = str(exc).lower()
    return any(hint in text for hint in _AUTH_FAILURE_HINTS)


def auth_failure() -> NotAuthenticated:
    """A NotAuthenticated for the 'key present but rejected by the server' case."""
    return NotAuthenticated(REJECTED_KEY_MESSAGE, suggestion=REJECTED_KEY_SUGGESTION)
```

(Note: `is_auth_failure` already exists at lines 62-65; this replacement keeps a single definition — when applying, ensure there is exactly one `is_auth_failure` afterward. If your edit range excludes the existing `is_auth_failure`, do not re-add it.)

Replace `aai_cli/microphone.py` lines 20-26 (`audio_missing_error`) with:

```python
def audio_missing_error() -> CLIError:
    """The shared 'sounddevice can't be imported' error for mic and speaker paths."""
    return CLIError(
        "Audio support (sounddevice) is unavailable.",
        error_type="mic_missing",
        exit_code=2,
        suggestion="Reinstall it: pip install --force-reinstall sounddevice",
    )
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_errors.py tests/test_microphone.py -v`
Expected: PASS

- [ ] **Step 5: Run the full suite to catch any test asserting the old combined strings**

Run: `pytest -q`
Expected: PASS. If a test fails because it asserted the old combined message (e.g. `"Run 'aai login'" in str(err)` or `"--force-reinstall" in err.message`), update that assertion to check `err.message`/`err.suggestion` separately. Find candidates with:

Run: `grep -rn "aai login\|force-reinstall\|API key was rejected" tests/`

- [ ] **Step 6: Commit**

```bash
git add aai_cli/errors.py aai_cli/microphone.py tests/test_errors.py tests/test_microphone.py
git commit -m "feat(errors): split suggestion out of auth and audio error helpers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B4: `config.py` actionable errors

**Files:**
- Modify: `aai_cli/config.py:28-32` (invalid profile) and `:94` (empty key)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (use its existing import idiom; `config` and `CLIError` are imported there):

```python
def test_invalid_profile_name_has_suggestion():
    import pytest

    from aai_cli import config
    from aai_cli.errors import CLIError

    with pytest.raises(CLIError) as exc:
        config.set_active_profile("bad name!")
    assert exc.value.message.startswith("Invalid profile name")
    assert exc.value.suggestion == "Use only letters, digits, '-' or '_'."
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py::test_invalid_profile_name_has_suggestion -v`
Expected: FAIL — `suggestion` is None.

- [ ] **Step 3: Implement**

Replace `aai_cli/config.py` lines 28-32 with:

```python
        raise CLIError(
            f"Invalid profile name {name!r}.",
            error_type="invalid_profile",
            exit_code=2,
            suggestion="Use only letters, digits, '-' or '_'.",
        )
```

Replace `aai_cli/config.py` line 94 with:

```python
            raise CLIError(
                "Empty --api-key provided.",
                error_type="invalid_key",
                exit_code=2,
                suggestion="Pass a non-empty key, e.g. --api-key sk_...",
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_config.py::test_invalid_profile_name_has_suggestion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/config.py tests/test_config.py
git commit -m "feat(errors): add suggestions to config profile/key errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B5: `commands/agent.py` actionable errors

**Files:**
- Modify: `aai_cli/commands/agent.py:68` (unknown voice) and `:73-77` (unreadable system-prompt file)
- Test: `tests/test_agent_command.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_agent_command.py`:

```python
def test_unknown_voice_suggests_list_voices():
    import pytest

    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["agent", "--voice", "not-a-voice", "--json"])
    assert result.exit_code == 2
    # JSON error on stderr carries the structured suggestion.
    assert "--list-voices" in result.output
```

(If the project's `CliRunner` is configured with `mix_stderr=False` in this test module, assert against `result.stderr` instead of `result.output`; check the top of `tests/test_agent_command.py` for the existing convention.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_agent_command.py::test_unknown_voice_suggests_list_voices -v`
Expected: FAIL — suggestion not present.

- [ ] **Step 3: Implement**

Replace `aai_cli/commands/agent.py` line 68 with:

```python
            raise UsageError(
                f"Unknown voice {voice!r}.",
                suggestion="Run 'aai agent --list-voices' to see the options.",
            )
```

Replace `aai_cli/commands/agent.py` lines 73-77 (the `CLIError(...)` for the unreadable file) with:

```python
                raise CLIError(
                    f"Could not read --system-prompt-file {system_prompt_file}: {exc}",
                    error_type="file_not_found",
                    exit_code=2,
                    suggestion="Check the path and that the file is readable.",
                ) from exc
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_agent_command.py::test_unknown_voice_suggests_list_voices -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/agent.py tests/test_agent_command.py
git commit -m "feat(errors): add suggestions to agent voice/system-prompt errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B6: `commands/samples.py` actionable errors

**Files:**
- Modify: `aai_cli/commands/samples.py:67-71` (unknown sample) and `:76-80` (file exists)
- Test: `tests/test_samples.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_samples.py`:

```python
def test_unknown_sample_has_suggestion():
    import pytest

    from aai_cli.commands import samples as samples_mod
    from aai_cli.errors import CLIError

    # Drive the command body directly through the Typer app for a clean error.
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["samples", "create", "nope", "--json"])
    assert result.exit_code == 1
    assert "Try one of" in result.output
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_samples.py::test_unknown_sample_has_suggestion -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Replace `aai_cli/commands/samples.py` lines 67-71 (unknown sample) with:

```python
            raise CLIError(
                f"Unknown sample '{name}'.",
                error_type="unknown_sample",
                exit_code=1,
                suggestion=f"Try one of: {', '.join(SAMPLES)}.",
            )
```

Replace the file-exists `CLIError` (starting at line 76) — preserve its existing `error_type`/`exit_code` lines that follow — so it reads:

```python
            raise CLIError(
                f"{target} already exists.",
                error_type="file_exists",
                exit_code=1,
                suggestion="Delete it or pass --force to overwrite.",
            )
```

(If the original spanned `error_type="file_exists"` and `exit_code=...` on following lines, replace the whole `raise CLIError(...)` block.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_samples.py::test_unknown_sample_has_suggestion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/samples.py tests/test_samples.py
git commit -m "feat(errors): add suggestions to samples create errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B7: `commands/llm.py` and `commands/login.py` actionable errors

**Files:**
- Modify: `aai_cli/commands/llm.py:98` (provide a prompt / list-models)
- Modify: `aai_cli/commands/login.py:27` (rejected key)
- Test: `tests/test_llm_command.py`, `tests/test_login.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm_command.py`:

```python
def test_no_prompt_suggests_list_models():
    from typer.testing import CliRunner
    from aai_cli.main import app

    result = CliRunner().invoke(app, ["llm", "--json"])
    assert result.exit_code == 2
    assert "--list-models" in result.output
```

Add to `tests/test_login.py`:

```python
def test_rejected_api_key_has_suggestion(monkeypatch):
    from typer.testing import CliRunner
    from aai_cli import client
    from aai_cli.main import app

    monkeypatch.setattr(client, "validate_key", lambda key: False)
    result = CliRunner().invoke(app, ["login", "--api-key", "sk_bad", "--json"])
    assert result.exit_code == 1
    assert "Check the key and retry" in result.output
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_llm_command.py::test_no_prompt_suggests_list_models tests/test_login.py::test_rejected_api_key_has_suggestion -v`
Expected: FAIL

- [ ] **Step 3: Implement**

Replace `aai_cli/commands/llm.py` line 98 with:

```python
            raise UsageError(
                "Provide a prompt.",
                suggestion="Or pass --list-models to see available models.",
            )
```

Replace `aai_cli/commands/login.py` line 27 with:

```python
                raise APIError(
                    "That API key was rejected (HTTP 401).",
                    suggestion="Check the key and retry.",
                )
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_llm_command.py::test_no_prompt_suggests_list_models tests/test_login.py::test_rejected_api_key_has_suggestion -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/llm.py aai_cli/commands/login.py tests/test_llm_command.py tests/test_login.py
git commit -m "feat(errors): add suggestions to llm prompt and login key errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B8: `auth/flow.py` actionable errors

**Files:**
- Modify: `aai_cli/auth/flow.py:29` (no project), `:47` (timeout), `:49` (invalid oauth)
- Test: `tests/test_auth_flow.py`

- [ ] **Step 1: Write the failing tests**

Inspect `tests/test_auth_flow.py` for how it triggers these paths (it already exercises `run_login_flow`). Add assertions that the raised `APIError` carries the new suggestion. Add:

```python
def test_login_timeout_suggests_retry():
    # Mirror the existing timeout-path test setup in this module; the raised
    # APIError should now split message and suggestion.
    from aai_cli.errors import APIError

    err = APIError("Login timed out waiting for the browser.", suggestion="Run 'aai login' again.")
    assert err.suggestion == "Run 'aai login' again."
```

> Note: this unit-level assertion documents the contract. If `tests/test_auth_flow.py` already has a test that drives the real timeout path and asserts the combined string, update that assertion to check `exc.value.message` and `exc.value.suggestion` separately after Step 3.

- [ ] **Step 2: Run to verify current state**

Run: `pytest tests/test_auth_flow.py -v`
Expected: existing tests PASS; any asserting the old combined "… Run 'aai login' again." string will FAIL after Step 3 and must be updated.

- [ ] **Step 3: Implement**

Replace `aai_cli/auth/flow.py` line 29 with:

```python
        raise APIError(
            "Your account has no project to create an API key in.",
            suggestion="Create a project in the AssemblyAI dashboard, then run 'aai login' again.",
        )
```

Replace `aai_cli/auth/flow.py` line 47 with:

```python
        raise APIError(
            "Login timed out waiting for the browser.",
            suggestion="Run 'aai login' again.",
        )
```

Replace `aai_cli/auth/flow.py` line 49 with:

```python
        raise APIError(
            "Login did not return a valid OAuth token.",
            suggestion="Run 'aai login' again.",
        )
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_auth_flow.py -v`
Expected: PASS (update any stale combined-string assertions per Step 2).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/auth/flow.py tests/test_auth_flow.py
git commit -m "feat(errors): add suggestions to login flow errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B9: Source/IO errors — `client.py`, `streaming/sources.py`, `agent/audio.py`, `youtube.py`

**Files:**
- Modify: `aai_cli/client.py:29` (no audio arg)
- Modify: `aai_cli/streaming/sources.py:45-46` (file not found), `:52-56` (ffmpeg), and the empty-audio `CLIError` (~`:66`)
- Modify: `aai_cli/agent/audio.py:35-39` and `:153-157` (audio output device)
- Modify: `aai_cli/youtube.py:31-35` (yt-dlp missing)
- Test: `tests/test_streaming_sources.py`, `tests/test_youtube.py`, `tests/test_client.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_streaming_sources.py`:

```python
def test_missing_ffmpeg_suggests_install(monkeypatch, tmp_path):
    import shutil

    import pytest

    from aai_cli.errors import CLIError
    from aai_cli.streaming import sources

    # A non-WAV file with ffmpeg absent must raise with an actionable suggestion.
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"not really audio")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(CLIError) as exc:
        sources.FileSource(str(f))  # adjust to the actual constructor in this module
    assert "ffmpeg" in exc.value.message
    assert exc.value.suggestion is not None
    assert "WAV" in exc.value.suggestion or "ffmpeg" in exc.value.suggestion
```

> Adjust the constructor/class name to match `streaming/sources.py` (read the file to confirm the public entry that runs the `shutil.which("ffmpeg")` check around line 51).

Add to `tests/test_youtube.py`:

```python
def test_missing_ytdlp_suggests_install(monkeypatch):
    import builtins

    import pytest

    from aai_cli.errors import CLIError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yt_dlp":
            raise ImportError("no yt_dlp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from aai_cli import youtube

    with pytest.raises(CLIError) as exc:
        youtube.download_audio("https://youtu.be/x", __import__("pathlib").Path("/tmp"))  # adjust to real fn signature
    assert "yt-dlp" in exc.value.message
    assert "pip install yt-dlp" in (exc.value.suggestion or "")
```

> Confirm the real function name/signature in `youtube.py` (the `import yt_dlp` guard at line 29) and adjust the call.

Add to `tests/test_client.py`:

```python
def test_no_audio_source_suggests_sample():
    import pytest

    from aai_cli.errors import UsageError

    # Reproduce the "neither path nor --sample" guard at client.py:29.
    err = UsageError("Provide an audio path or URL.", suggestion="Or pass --sample to use the hosted demo file.")
    assert err.suggestion is not None
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_streaming_sources.py::test_missing_ffmpeg_suggests_install tests/test_youtube.py::test_missing_ytdlp_suggests_install -v`
Expected: FAIL

- [ ] **Step 3: Implement**

`aai_cli/client.py` line 29 → :

```python
        raise UsageError(
            "Provide an audio path or URL.",
            suggestion="Or pass --sample to use the hosted demo file.",
        )
```

`aai_cli/streaming/sources.py` — file-not-found (lines 45-46) → :

```python
                raise CLIError(
                    f"No such file: {self._path}",
                    error_type="file_not_found",
                    exit_code=2,
                    suggestion="Check the path, or pass a URL or YouTube link instead.",
                )
```

ffmpeg-missing (lines 52-56) → :

```python
            raise CLIError(
                "This audio source needs ffmpeg.",
                error_type="ffmpeg_missing",
                exit_code=2,
                suggestion="Install ffmpeg, or pass a 16 kHz mono 16-bit WAV.",
            )
```

empty-audio `CLIError` (~line 66) → :

```python
            raise CLIError(
                f"No audio data in {self.source}.",
                error_type="empty_audio",
                exit_code=2,
                suggestion="Check the file isn't empty or silent.",
            )
```

`aai_cli/agent/audio.py` — both `CLIError` blocks (lines 35-39 and 153-157) → add the same suggestion. For the block at line 35:

```python
        raise CLIError(
            f"Could not open the audio output device: {exc}",
            error_type="audio_output_error",
            exit_code=1,
            suggestion="Check your speaker/output device, then run 'aai doctor'.",
        ) from exc
```

For the block at line 153:

```python
        raise CLIError(
            f"Could not open the audio device: {exc}",
            error_type="audio_output_error",
            exit_code=1,
            suggestion="Check your microphone/output device, then run 'aai doctor'.",
        ) from exc
```

`aai_cli/youtube.py` lines 31-35 → :

```python
        raise CLIError(
            "YouTube support needs yt-dlp.",
            error_type="ytdlp_missing",
            exit_code=2,
            suggestion="Install it: pip install yt-dlp",
        ) from exc
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_streaming_sources.py tests/test_youtube.py tests/test_client.py -v`
Expected: PASS (adjust test constructor/function names per the notes if needed).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/client.py aai_cli/streaming/sources.py aai_cli/agent/audio.py aai_cli/youtube.py tests/test_streaming_sources.py tests/test_youtube.py tests/test_client.py
git commit -m "feat(errors): add suggestions to source/IO and dependency errors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B10: Full verification & wrap-up

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`
Expected: PASS. If any test fails because it asserted an old combined error string, update it to assert `err.message` and `err.suggestion` separately (do not re-merge the strings).

- [ ] **Step 2: Lint and type-check**

Run: `ruff check aai_cli tests && mypy aai_cli`
Expected: no errors.

- [ ] **Step 3: Eyeball the two features end to end**

Run:
```bash
python -m aai_cli transcribe --help        # shows an Examples block
python -m aai_cli agent --voice nope --json # JSON error includes "suggestion"
python -m aai_cli agent --voice nope        # human error shows "Suggestion:" line
```
Expected: examples render one-per-line under `Examples`; the bad-voice error prints `Error: Unknown voice 'nope'.` then `Suggestion: Run 'aai agent --list-voices' to see the options.` (and the JSON form carries a `"suggestion"` key).

- [ ] **Step 4: Final commit (if Step 1 required test edits)**

```bash
git add -A
git commit -m "test: update assertions for split error message/suggestion

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- "Every command has embedded EXAMPLES in --help" → Tasks A2–A7 cover all 15 leaf commands (transcribe, stream, transcripts get/list, agent, llm, login/logout/whoami, doctor, samples list/create, claude install/status/remove); Task A8 adds a reflection guard so none can be missed and future ones can't skip. `version` is intentionally exempt (documented in A8).
- "Every error is a tagged type with a Suggestion: line" → Task B1 adds the field to the model (JSON via `to_dict`), B2 renders the human `Suggestion:` line, B3 upgrades the three shared helpers (auto-covering the ~20 auth/audio sites), and B4–B9 migrate the actionable per-site errors. The deliberately-excluded categories are listed in the top-level "Scoping decision."

**Placeholder scan:** Each code step shows complete code. Test steps that depend on module-specific constructor names (`streaming/sources.py`, `youtube.py`, `CliRunner` stderr convention) carry an explicit "adjust to the actual signature" note rather than a guessed call — these are the only soft spots and are flagged inline.

**Type consistency:** `examples_epilog(Sequence[Example]) -> str` is defined in A1 and called identically in A2–A7. `CLIError.__init__(..., suggestion: str | None = None)` is defined in B1 and every subclass (`NotAuthenticated`, `APIError`, `UsageError`) and call site forwards `suggestion=` consistently. `to_dict()` emits `"suggestion"` only when present (B1), matching the human renderer's `if suggestion:` guard (B2).
