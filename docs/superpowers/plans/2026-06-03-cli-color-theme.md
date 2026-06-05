# CLI Color Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `aai` one centralized Rich color theme (AssemblyAI brand accent + semantic styles) applied consistently to all human-facing output.

**Architecture:** A new `assemblyai_cli/theme.py` owns a single `rich.theme.Theme` of semantic style names (`aai.brand`, `aai.success`, …) plus helpers (`make_console`, `speaker_style`, `status_style`). Every `Console` is built through `make_console` so the style names resolve globally. Call sites switch from ad-hoc `[red]`/`[green]` markup to the semantic names, and the streaming/agent renderers color role/speaker labels via styled `rich.text.Text`. Color only reaches interactive TTYs (Rich auto-disables ANSI on pipes/non-tty and honors `NO_COLOR`); the agentic/CI path already routes to JSON.

**Tech Stack:** Python, Typer, Rich 15, pytest (90% branch-coverage gate), ruff, mypy.

---

## File Structure

- **Create** `assemblyai_cli/theme.py` — the palette, `THEME`, `make_console`, `speaker_style`, `status_style`.
- **Create** `tests/test_theme.py` — unit tests for the theme module.
- **Modify** `assemblyai_cli/output.py` — themed shared console + themed error markup.
- **Modify** `assemblyai_cli/render.py` — `BaseRenderer` line helpers accept `str | Text`; themed per-stream console; `stopped()` muted.
- **Modify** `assemblyai_cli/streaming/render.py` — muted lifecycle notice, brand LLM line.
- **Modify** `assemblyai_cli/agent/render.py` — muted notices, accent `you:`/`agent:` labels.
- **Modify** `assemblyai_cli/commands/transcribe.py` — accent speaker labels.
- **Modify** `assemblyai_cli/commands/transcripts.py` — brand table header + status-colored cell.
- **Modify** `assemblyai_cli/commands/claude.py` — status-colored steps + brand heading.
- **Modify** `assemblyai_cli/commands/login.py` and `assemblyai_cli/commands/samples.py` — swap raw color markup for semantic names.
- **Modify** `tests/test_streaming_render.py`, `tests/test_agent_render.py` — `_human()` helper builds a themed console (so `aai.*` style names resolve); add color assertions.

Notes for the implementer:
- ruff lint `select` does NOT include `ANN`, so `**kwargs: Any` is allowed. mypy has `disallow_untyped_defs=true` for src, relaxed for tests.
- A Rich `Console` only resolves a style **name** like `"aai.label"` if the theme is attached; otherwise it raises `rich.errors.MissingStyle` at render time. This is why test consoles must be built via `make_console`.
- `color_system=None` still resolves theme style names; it just emits no ANSI. Use `color_system="truecolor"` in tests that assert color escape codes are present.

---

## Task 1: Theme module

**Files:**
- Create: `assemblyai_cli/theme.py`
- Test: `tests/test_theme.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_theme.py`:

```python
import io

from assemblyai_cli import theme


def test_make_console_resolves_named_styles():
    console = theme.make_console()
    # get_style raises rich.errors.MissingStyle if a name is not in the theme.
    for name in (
        "aai.brand",
        "aai.heading",
        "aai.label",
        "aai.success",
        "aai.error",
        "aai.warn",
        "aai.muted",
    ):
        console.get_style(name)
    for name in theme.SPEAKER_STYLES:
        console.get_style(name)


def test_make_console_passes_kwargs_through():
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, width=42)
    assert console.file is buf
    assert console.width == 42


def test_status_style_maps_known_statuses():
    assert theme.status_style("completed") == "aai.success"
    assert theme.status_style("ERROR") == "aai.error"
    assert theme.status_style("failed") == "aai.error"
    assert theme.status_style("queued") == "aai.warn"
    assert theme.status_style("processing") == "aai.warn"


def test_status_style_unknown_falls_back_to_muted():
    assert theme.status_style("something-else") == "aai.muted"


def test_speaker_style_deterministic_and_in_palette():
    assert theme.speaker_style("A") in theme.SPEAKER_STYLES
    assert theme.speaker_style("A") == theme.speaker_style("A")
    assert theme.speaker_style("A") != theme.speaker_style("B")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_theme.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'assemblyai_cli.theme'`.

- [ ] **Step 3: Write the implementation**

Create `assemblyai_cli/theme.py`:

```python
from __future__ import annotations

from typing import IO, Any

from rich.console import Console
from rich.theme import Theme

# AssemblyAI brand accent. Defined once so the whole CLI can be re-tinted here.
BRAND = "#2545D3"

# Per-speaker label colors, rotated deterministically by speaker_style().
SPEAKER_STYLES: tuple[str, ...] = (
    "aai.speaker.0",
    "aai.speaker.1",
    "aai.speaker.2",
    "aai.speaker.3",
    "aai.speaker.4",
)

THEME = Theme(
    {
        "aai.brand": f"bold {BRAND}",
        "aai.heading": f"bold {BRAND}",
        "aai.label": BRAND,
        "aai.success": "green",
        "aai.error": "bold red",
        "aai.warn": "yellow",
        "aai.muted": "dim",
        "aai.speaker.0": BRAND,
        "aai.speaker.1": "cyan",
        "aai.speaker.2": "magenta",
        "aai.speaker.3": "green",
        "aai.speaker.4": "yellow",
    }
)

# Status strings grouped by the semantic style they render in.
_SUCCESS = {"completed", "installed", "removed", "ok", "present", "authenticated"}
_ERROR = {"error", "failed"}
_WARN = {"queued", "processing", "in_progress", "running"}


def make_console(file: IO[str] | None = None, **kwargs: Any) -> Console:
    """Build a Console with the AssemblyAI theme attached so `aai.*` names resolve."""
    return Console(file=file, theme=THEME, **kwargs)


def speaker_style(speaker: object) -> str:
    """Deterministically map a speaker id to one of SPEAKER_STYLES."""
    key = str(speaker)
    idx = sum(ord(c) for c in key) % len(SPEAKER_STYLES)
    return SPEAKER_STYLES[idx]


def status_style(status: str) -> str:
    """Map a transcript/setup status to a semantic style name (muted if unknown)."""
    normalized = status.strip().lower()
    if normalized in _SUCCESS:
        return "aai.success"
    if normalized in _ERROR:
        return "aai.error"
    if normalized in _WARN:
        return "aai.warn"
    return "aai.muted"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_theme.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/theme.py tests/test_theme.py && ruff format --check assemblyai_cli/theme.py tests/test_theme.py && mypy assemblyai_cli/theme.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/theme.py tests/test_theme.py
git commit -m "feat(theme): add centralized Rich color theme module"
```

---

## Task 2: Route the shared console through the theme

**Files:**
- Modify: `assemblyai_cli/output.py`
- Test: `tests/test_theme.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_theme.py`:

```python
def test_output_console_is_themed_and_error_is_styled(monkeypatch):
    from assemblyai_cli import output, theme
    from assemblyai_cli.errors import CLIError

    buf = io.StringIO()
    monkeypatch.setattr(
        output,
        "console",
        theme.make_console(file=buf, force_terminal=True, color_system="truecolor"),
    )
    output.emit_error(CLIError("boom"), json_mode=False)
    out = buf.getvalue()
    assert "Error:" in out
    assert "boom" in out
    assert "\x1b[" in out  # themed error emits ANSI on a forced-color console
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_theme.py::test_output_console_is_themed_and_error_is_styled -v`
Expected: FAIL — the current `output.console` has no theme, so rendering `[aai.error]` raises `MissingStyle` (or the ANSI assertion fails).

- [ ] **Step 3: Edit `assemblyai_cli/output.py`**

Add the import near the other `assemblyai_cli` imports (keep `TYPE_CHECKING` block as-is):

```python
from assemblyai_cli import theme
```

Replace line 17 `console = Console()` with:

```python
console = theme.make_console()
```

The unused `Console` import can stay only if still referenced; it is not, so remove `from rich.console import Console`. Replace the error line (was line 48):

```python
        console.print(f"[aai.error]Error:[/aai.error] {escape(err.message)}")
```

- [ ] **Step 4: Run the test + full suite**

Run: `pytest tests/test_theme.py -v && pytest -q -m "not e2e"`
Expected: PASS, no regressions.

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/output.py && mypy assemblyai_cli/output.py`
Expected: no errors (no leftover unused `Console` import).

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/output.py tests/test_theme.py
git commit -m "feat(theme): theme the shared output console and error markup"
```

---

## Task 3: BaseRenderer accepts styled Text + themed per-stream console

**Files:**
- Modify: `assemblyai_cli/render.py`
- Test: `tests/test_streaming_render.py`

- [ ] **Step 1: Update the test helper and add a coverage test**

In `tests/test_streaming_render.py`, change the imports and `_human` helper. Replace:

```python
from rich.console import Console

from assemblyai_cli.streaming.render import StreamRenderer
```

with:

```python
from assemblyai_cli import theme
from assemblyai_cli.streaming.render import StreamRenderer
```

Replace the `_human` helper body's console line. The helper becomes:

```python
def _human(width=80, color_system=None):
    """A human-mode renderer writing to a forced-terminal themed console buffer."""
    buf = io.StringIO()
    console = theme.make_console(
        file=buf, force_terminal=True, width=width, color_system=color_system
    )
    return StreamRenderer(json_mode=False, out=buf, console=console), buf
```

Add a test that the default per-stream console (no explicit console passed) is themed:

```python
def test_default_console_is_themed():
    buf = io.StringIO()
    r = StreamRenderer(json_mode=False, out=buf)
    # _console_obj builds via theme.make_console, so aai.* names resolve.
    r._console_obj().get_style("aai.brand")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_streaming_render.py::test_default_console_is_themed -v`
Expected: FAIL — `BaseRenderer._console_obj` currently builds a bare `Console`, so `get_style("aai.brand")` raises `MissingStyle`.

- [ ] **Step 3: Edit `assemblyai_cli/render.py`**

Add the import after the existing imports (keep the rich imports):

```python
from assemblyai_cli import theme
```

Change `_console_obj` (was lines 45-48) to build via the theme:

```python
    def _console_obj(self) -> Console:
        if self._console is None:
            self._console = theme.make_console(file=self.out)
        return self._console
```

Change the three line helpers so they accept `str | Text`. Replace `_update_line`, `_finalize_line`, and `_line` (was lines 70-86) with:

```python
    @staticmethod
    def _as_text(text: str | Text) -> Text:
        return text if isinstance(text, Text) else Text(text)

    def _update_line(self, text: str | Text) -> None:
        """Redraw the in-progress line in place (Rich clears any prior wrap)."""
        self._live_obj().update(self._as_text(text), refresh=True)

    def _finalize_line(self, text: str | Text | None = None) -> None:
        """Commit the in-progress line (optionally replacing its text) as permanent."""
        if self._live is not None:
            if text is not None:
                self._live.update(self._as_text(text), refresh=True)
            self._commit_live()
        elif text is not None:
            self._console_obj().print(self._as_text(text))

    def _line(self, text: str | Text) -> None:
        """Print a standalone permanent line, committing any open partial first."""
        self._commit_live()
        self._console_obj().print(self._as_text(text))
```

Change `stopped()` (was lines 89-91) to render muted:

```python
    def stopped(self) -> None:
        if not self.json_mode:
            self._line(Text("Stopped.", style="aai.muted"))
```

- [ ] **Step 4: Run the streaming + agent render suites**

Run: `pytest tests/test_streaming_render.py tests/test_agent_render.py -v`
Expected: PASS. (The agent suite still uses its own bare-console `_human`; it is updated in Task 5. The named styles used so far only appear in StreamRenderer's themed console, so agent tests are unaffected here.)

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/render.py && mypy assemblyai_cli/render.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/render.py tests/test_streaming_render.py
git commit -m "feat(theme): themed base renderer console and styled line helpers"
```

---

## Task 4: StreamRenderer styling

**Files:**
- Modify: `assemblyai_cli/streaming/render.py`
- Test: `tests/test_streaming_render.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streaming_render.py`:

```python
def test_human_begin_notice_is_muted():
    r, buf = _human(color_system="truecolor")
    r.begin(types.SimpleNamespace(id="x"))
    assert "\x1b[" in buf.getvalue()  # muted styling emits ANSI


def test_human_llm_line_is_branded():
    r, buf = _human(color_system="truecolor")
    r.turn(_turn("hola", True))
    r.llm("the summary")
    out = buf.getvalue()
    assert "the summary" in out
    assert "\x1b[" in out  # brand styling emits ANSI
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_streaming_render.py::test_human_begin_notice_is_muted tests/test_streaming_render.py::test_human_llm_line_is_branded -v`
Expected: FAIL — the current `begin`/`llm` pass plain strings; with `color_system="truecolor"` but no styling, no ANSI is emitted.

- [ ] **Step 3: Edit `assemblyai_cli/streaming/render.py`**

Add the import:

```python
from rich.text import Text

from assemblyai_cli import theme
from assemblyai_cli.render import BaseRenderer
```

Change the `begin` human branch (was line 13):

```python
            self._line(Text("Listening… (Ctrl-C to stop)", style="aai.muted"))
```

Change the `llm` human branch (was line 41):

```python
            self._line(Text("\N{ELECTRIC LIGHT BULB} " + content, style="aai.brand"))
```

- [ ] **Step 4: Run the streaming suite**

Run: `pytest tests/test_streaming_render.py -v`
Expected: PASS (existing `test_human_begin_prints_notice`, `test_human_llm_line_rendered` still pass — text content is unchanged).

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/streaming/render.py && mypy assemblyai_cli/streaming/render.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/streaming/render.py tests/test_streaming_render.py
git commit -m "feat(theme): color streaming notice and LLM line"
```

---

## Task 5: AgentRenderer styling (role labels)

**Files:**
- Modify: `assemblyai_cli/agent/render.py`
- Test: `tests/test_agent_render.py`

- [ ] **Step 1: Update the test helper and add color tests**

In `tests/test_agent_render.py`, replace:

```python
from rich.console import Console

from assemblyai_cli.agent.render import AgentRenderer
```

with:

```python
from assemblyai_cli import theme
from assemblyai_cli.agent.render import AgentRenderer
```

Replace the `_human` helper:

```python
def _human(width=80, color_system=None):
    """A human-mode renderer writing to a forced-terminal themed console buffer."""
    buf = io.StringIO()
    console = theme.make_console(
        file=buf, force_terminal=True, width=width, color_system=color_system
    )
    return AgentRenderer(json_mode=False, out=buf, console=console), buf
```

Add:

```python
def test_human_agent_label_is_colored():
    r, buf = _human(color_system="truecolor")
    r.agent_transcript("the time is noon", interrupted=False)
    out = buf.getvalue()
    assert "agent: " in out
    assert "the time is noon" in out
    assert "\x1b[" in out  # label styling emits ANSI


def test_human_you_label_is_colored():
    r, buf = _human(color_system="truecolor")
    r.user_final("what is the time")
    r.close()
    out = buf.getvalue()
    assert "you: " in out
    assert "what is the time" in out
    assert "\x1b[" in out
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_agent_render.py::test_human_agent_label_is_colored tests/test_agent_render.py::test_human_you_label_is_colored -v`
Expected: FAIL — labels are currently plain text, so no ANSI is emitted even with color forced.

- [ ] **Step 3: Edit `assemblyai_cli/agent/render.py`**

Add imports:

```python
from rich.text import Text

from assemblyai_cli import theme
from assemblyai_cli.render import BaseRenderer
```

Add a small helper at module scope (after the imports, before the class) for the `label: body` pattern:

```python
def _labeled(label: str, body: str) -> Text:
    """A line whose `label` prefix is brand-accented and whose body is default."""
    return Text.assemble((label, "aai.label"), body)
```

Change `connected` human branch (was line 17):

```python
            self._line(Text("Connected — start talking. (Ctrl-C to stop)", style="aai.muted"))
```

Change `user_partial` human branch (was line 28):

```python
        self._update_line(_labeled("you: ", text))
```

Change `user_final` human branch (was line 34):

```python
        self._finalize_line(_labeled("you: ", text))
```

Change `agent_transcript` human branch (was line 45):

```python
        self._line(_labeled("agent: ", text))  # commits any open "you: …" partial first
```

- [ ] **Step 4: Run the agent suite**

Run: `pytest tests/test_agent_render.py -v`
Expected: PASS (existing `test_human_agent_line_labeled`, `test_human_partial_then_final`, `test_human_connected_and_stopped_announce` still pass — text is unchanged).

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/agent/render.py && mypy assemblyai_cli/agent/render.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/agent/render.py tests/test_agent_render.py
git commit -m "feat(theme): accent you:/agent: labels and mute lifecycle notices"
```

---

## Task 6: Transcribe speaker-label coloring

**Files:**
- Modify: `assemblyai_cli/commands/transcribe.py`
- Test: `tests/test_transcribe.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcribe.py` (top-of-file imports as needed):

```python
def test_render_transcript_colors_speaker_labels():
    import io

    from assemblyai_cli import theme
    from assemblyai_cli.commands.transcribe import _render_transcript

    data = {
        "text": "ignored when utterances present",
        "utterances": [
            {"speaker": "A", "text": "hello", "start": 0, "end": 1},
            {"speaker": "B", "text": "hi there", "start": 1, "end": 2},
        ],
    }
    rendered = _render_transcript(data)
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, color_system="truecolor")
    console.print(rendered)
    out = buf.getvalue()
    assert "Speaker A:" in out
    assert "hello" in out
    assert "Speaker B:" in out
    assert "\x1b[" in out  # speaker labels are styled


def test_render_transcript_plain_text_unchanged():
    from assemblyai_cli.commands.transcribe import _render_transcript

    assert _render_transcript({"text": "just the words"}) == "just the words"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_transcribe.py::test_render_transcript_colors_speaker_labels -v`
Expected: FAIL — `_render_transcript` currently returns a plain escaped string with no styling, so no ANSI is emitted.

- [ ] **Step 3: Edit `assemblyai_cli/commands/transcribe.py`**

Add imports (alongside `from rich.markup import escape`):

```python
from rich.text import Text

from assemblyai_cli import client, config, llm, output, theme, youtube
```

(Extend the existing `from assemblyai_cli import ...` line to include `theme`.)

Replace `_render_transcript` (was lines 23-29) with the version below. Note: the
existing line 27 carries a stale `# type: ignore[union-attr]` that whole-project
mypy now rejects (`utterances` is typed `object`, so the real error is
`attr-defined`/`unused-ignore`). Since we are rewriting this exact function, fix
it properly by narrowing with `isinstance(..., list)` — no `type: ignore` needed,
and mypy passes cleanly:

```python
def _render_transcript(data: dict[str, object]) -> str | Text:
    """Human view: speaker-labeled lines when diarized, otherwise the plain text."""
    utterances = data.get("utterances")
    if isinstance(utterances, list) and utterances:
        line = Text()
        for i, u in enumerate(utterances):
            if i:
                line.append("\n")
            line.append(f"Speaker {u['speaker']}: ", style=theme.speaker_style(u["speaker"]))
            line.append(str(u["text"]))
        return line
    return escape(str(data["text"]))
```

(`isinstance(utterances, list)` narrows `object` → `list[Any]`, so `u` is `Any`
and `u['speaker']` / `u['text']` type-check without an ignore. The `and utterances`
keeps the original behavior of falling through to plain text on an empty list.)

- [ ] **Step 4: Run the transcribe suite**

Run: `pytest tests/test_transcribe.py -v`
Expected: PASS. (The non-diarized path still returns the escaped string, so existing assertions hold.)

- [ ] **Step 5: Lint/typecheck**

The file has pre-existing format drift, so run the formatter on it (we are
legitimately touching the file), then lint and typecheck:

Run: `ruff format assemblyai_cli/commands/transcribe.py && ruff check assemblyai_cli/commands/transcribe.py && mypy`
Expected: ruff clean; **whole-project `mypy` now reports "no issues"** — the 2 pre-existing
`transcribe.py:27` errors are resolved by the `isinstance` narrowing. (`output.emit`'s
renderer param is `Callable[[T], object]`, so a `str | Text` return is fine.)

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/commands/transcribe.py tests/test_transcribe.py
git commit -m "feat(theme): color diarized speaker labels in transcribe output"
```

---

## Task 7: Transcripts table — header + status coloring

**Files:**
- Modify: `assemblyai_cli/commands/transcripts.py`
- Test: `tests/test_transcripts.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transcripts.py`:

```python
def test_list_table_colors_status(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("assemblyai_cli.output.resolve_json", lambda *, explicit: False)
    # Force a real color terminal so styling produces ANSI we can assert on.
    monkeypatch.setattr(
        "assemblyai_cli.output.console",
        __import__("assemblyai_cli.theme", fromlist=["make_console"]).make_console(
            force_terminal=True, color_system="truecolor"
        ),
    )
    rows = [{"id": "t1", "status": "completed", "created": "2026-01-01"}]
    with patch("assemblyai_cli.commands.transcripts.client.list_transcripts", return_value=rows):
        result = runner.invoke(app, ["transcripts", "list"], color=True)
    assert result.exit_code == 0
    assert "completed" in result.output
    assert "\x1b[" in result.output  # status cell is colored
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_transcripts.py::test_list_table_colors_status -v`
Expected: FAIL — the table currently renders the status as an unstyled string, so no ANSI appears.

- [ ] **Step 3: Edit `assemblyai_cli/commands/transcripts.py`**

Add imports:

```python
from rich.text import Text

from assemblyai_cli import client, config, output, theme
```

(Extend the existing `from assemblyai_cli import ...` line to include `theme`.)

Replace the `render` closure (was lines 55-63) with:

```python
        def render(data: list[dict[str, object]]) -> Table:
            table = Table("id", "status", "created", header_style="aai.heading")
            for row in data:
                status = str(row["status"])
                table.add_row(
                    escape(str(row["id"])),
                    Text(status, style=theme.status_style(status)),
                    escape(str(row.get("created", ""))),
                )
            return table
```

- [ ] **Step 4: Run the transcripts suite**

Run: `pytest tests/test_transcripts.py -v`
Expected: PASS (existing `test_list_human_mode_renders_table` and `--json` tests still pass — JSON path is untouched, and table still contains the id/status text).

- [ ] **Step 5: Lint/typecheck**

Run: `ruff check assemblyai_cli/commands/transcripts.py && mypy assemblyai_cli/commands/transcripts.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add assemblyai_cli/commands/transcripts.py tests/test_transcripts.py
git commit -m "feat(theme): brand table header and status-colored cells in transcripts list"
```

---

## Task 8: Claude steps + login/samples semantic markup

**Files:**
- Modify: `assemblyai_cli/commands/claude.py`, `assemblyai_cli/commands/login.py`, `assemblyai_cli/commands/samples.py`
- Test: `tests/test_agent_command.py` (claude install/status), plus existing login/samples tests

- [ ] **Step 1: Write the failing test**

Add a focused unit test for the steps renderer. Create `tests/test_claude_render.py`:

```python
import io

from assemblyai_cli import theme
from assemblyai_cli.commands.claude import _render_steps


def test_render_steps_colors_status():
    data = {
        "steps": [
            {"name": "mcp", "status": "installed", "detail": "/path"},
            {"name": "skill", "status": "failed", "detail": "nope"},
        ]
    }
    rendered = _render_steps(data)
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, color_system="truecolor")
    console.print(rendered)
    out = buf.getvalue()
    assert "installed" in out
    assert "failed" in out
    assert "\x1b[" in out  # statuses are colored
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_claude_render.py -v`
Expected: FAIL — `_render_steps` returns a plain string today; no ANSI is emitted.

- [ ] **Step 3: Edit `assemblyai_cli/commands/claude.py`**

Ensure `theme` is imported (extend the existing `from assemblyai_cli import ...`):

```python
from assemblyai_cli import ... , theme
```

Replace `_render_steps` (was lines 185-187) with a version that styles each status and the heading via markup against the theme:

```python
def _render_steps(data: dict[str, list[Step]]) -> str:
    lines = []
    for s in data["steps"]:
        style = theme.status_style(s["status"])
        lines.append(
            f"  {escape(s['name'])}: "
            f"[{style}]{escape(s['status'])}[/{style}] — {escape(s['detail'])}"
        )
    return "[aai.heading]AssemblyAI coding-agent setup:[/aai.heading]\n" + "\n".join(lines)
```

(`escape` is already imported in `claude.py`. The returned markup string is rendered by `output.emit` through the themed `output.console`.)

- [ ] **Step 4: Edit `assemblyai_cli/commands/login.py`**

Replace the two color markups. The `[green]Authenticated[/green]` (was line 44):

```python
            lambda _d: f"[aai.success]Authenticated[/aai.success] on profile '{escape(profile)}'.",
```

The `[dim]…[/dim]` browser-fallback notice (was lines 35-37):

```python
                output.console.print(
                    "[aai.muted]Could not open a browser; open the URL above manually.[/aai.muted]"
                )
```

- [ ] **Step 5: Edit `assemblyai_cli/commands/samples.py`**

Replace `[yellow]Note:[/yellow]` (was line 85):

```python
                f"[aai.warn]Note:[/aai.warn] this file contains your API key — do not commit it.\n"
```

- [ ] **Step 6: Run the affected suites + full suite**

Run: `pytest tests/test_claude_render.py tests/test_agent_command.py -v && pytest -q -m "not e2e"`
Expected: PASS, no regressions. (Login/samples human-mode tests run on non-tty capture, so the themed markup renders as plain text and existing `in` assertions hold.)

- [ ] **Step 7: Lint/typecheck**

Run: `ruff check assemblyai_cli/commands/claude.py assemblyai_cli/commands/login.py assemblyai_cli/commands/samples.py tests/test_claude_render.py && mypy assemblyai_cli/commands/claude.py assemblyai_cli/commands/login.py assemblyai_cli/commands/samples.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add assemblyai_cli/commands/claude.py assemblyai_cli/commands/login.py assemblyai_cli/commands/samples.py tests/test_claude_render.py
git commit -m "feat(theme): semantic colors for setup steps, login, and samples notices"
```

---

## Task 9: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole check script**

Run: `bash scripts/check.sh`
Expected for the color work: `ruff check`, `mypy`, and `pytest` (`--cov-fail-under=90`) pass.

**Known pre-existing failures NOT caused by this work (do not fix — out of scope):**
`ruff format --check` reports drift in `assemblyai_cli/agent/audio.py`,
`assemblyai_cli/commands/stream.py`, and `tests/test_microphone.py` — these are
uncommitted in-flight changes that predate the color work and are owned by other
branch work. (`commands/transcribe.py` is fixed by Task 6.) If `check.sh` halts on
`ruff format --check` before reaching pytest, verify the color work independently:
`ruff check . && mypy && pytest -q -m "not e2e" --cov=assemblyai_cli --cov-branch --cov-fail-under=90`,
and report the pre-existing format drift separately rather than reformatting files
outside this plan.

- [ ] **Step 2: Manual smoke (optional, interactive TTY)**

Run in a real terminal: `aai transcripts list` and `aai --help`-driven flows; confirm brand-blue headers, green/red statuses, and accent `you:`/`agent:` labels appear, and that `NO_COLOR=1 aai transcripts list` and `aai transcripts list | cat` emit no ANSI.

- [ ] **Step 3: Commit any coverage-driven test additions**

If `--cov-fail-under=90` flags an uncovered branch (e.g. an untested `status_style` group), add a targeted test and commit:

```bash
git add tests/
git commit -m "test(theme): cover remaining theme branches"
```

---

## Self-Review

- **Spec coverage:**
  - `theme.py` with `BRAND`, `THEME`, semantic names, `SPEAKER_STYLES`, `make_console`, `speaker_style`, `status_style` → Task 1. ✓
  - Route all consoles through theme (`output.console`, `BaseRenderer`) → Tasks 2, 3. ✓
  - `emit_error` themed → Task 2. ✓
  - BaseRenderer accepts `str | Text`, `stopped()` muted → Task 3. ✓
  - StreamRenderer notice muted, LLM line brand → Task 4. ✓
  - AgentRenderer `you:`/`agent:` accent, notices muted → Task 5. ✓
  - Transcribe speaker labels rotating accent, plain path unchanged → Task 6. ✓
  - Transcripts table brand header + status-colored cell → Task 7. ✓
  - Claude steps status-colored + heading; login/samples semantic markup → Task 8. ✓
  - No-regression / coverage gate / NO_COLOR + pipe behavior → Task 9. ✓
  - `llm.py` needs no code change (uses themed `output.console` automatically) — noted in spec. ✓
- **Placeholder scan:** none — every code step shows full code; every run step shows the command and expected result.
- **Type consistency:** `make_console(file, **kwargs)`, `speaker_style(speaker)`, `status_style(status)`, and the `aai.*` style names are used identically across Tasks 1–8. `_render_transcript` and `_render_steps` return types (`str | Text`, `str`) match how `output.emit`/`console.print` consume them.
