# `assembly live` Tool-Call UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `assembly live` tool-call lines show their identifying argument and sit with a blank line above the block, while staying tight between consecutive calls.

**Architecture:** Two disjoint changes. (1) `brain.py` composes the friendly tool label with its identifying arg via the existing `describe_args()` helper, so the `Renderer.tool_call(label)` string gets richer with no protocol change. (2) A new dim `ToolAffordance` transcript widget (in `code_agent/messages.py`) replaces the shared `Note` for live tool calls, and `LiveAgentApp` styles it so the first call of a turn carries a top margin and consecutive ones are tight.

**Tech Stack:** Python 3.12+, Typer/Textual TUI, pytest + syrupy + pytest-textual-snapshot, `uv`.

## Global Constraints

- `from __future__ import annotations` at the top of every module (already present in all files touched).
- Spec: `docs/superpowers/specs/2026-06-22-live-tool-call-ux-design.md`.
- **Light touch only.** Do NOT add tool results, spinners, completion states, or expand `_TOOL_LABELS`. Do NOT touch the `Renderer` protocol, `engine.py`, `AgentRenderer`, or `_tool_label`.
- **Separator is `" · "`** (the middle dot, matching the live footer's `·` style). Trailing `…` is appended by the renderers, never by `_tool_affordance`.
- **Mutation gate is diff-scoped (vs `origin/main`) and requires assertions that fail if the changed line breaks** — not just coverage. Every boolean/branch added below is killed by a test asserting the *behavioral* difference between its two values.
- **Snapshots are regenerated, never hand-edited:** `uv run pytest tests/test_tui_snapshots.py --snapshot-update`, then eyeball the changed SVG before committing.
- **Commit hook:** a PreToolUse hook blocks `git commit` unless `./scripts/check.sh` last passed for the current working-tree signature. Use `AAI_ALLOW_COMMIT=1 git commit …` for the per-task WIP commits below, then run the **full** `./scripts/check.sh` once at the end (Task 4) and let that gate the final state.
- **Workspace isolation:** the working tree has unrelated in-flight work (push-to-talk/mute, model swap to `kimi-k2.5`, interrupt logic) touching `agent_cascade/tui.py`, `engine.py`, `config.py`. This feature is disjoint from it (different functions/lines; `messages.py` is untouched by the in-flight work). Execute on the `live-tool-call-ux` branch already created; commit ONLY this feature's files (`brain.py`, `messages.py`, `tui.py`, the three test files, the snapshot golden). Never `git add -A`.

---

### Task 1: Compose the tool label with its identifying argument (`brain.py`)

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (imports near line 21–26; add `_tool_affordance` after `_tool_label` at line 52; change the `on_tool(...)` call at line 285)
- Test: `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Consumes: `_tool_label(name: str) -> str` (existing, `brain.py:50`); `describe_args(args: Mapping[str, object]) -> str` (existing, `aai_cli/code_agent/summarize.py:36` — returns the one identifying arg clipped to 60 chars, or `""` when there are no args); `events.ToolCall` has `.name: str` and `.args: dict[str, object]`.
- Produces: `_tool_affordance(name: str, args: Mapping[str, object]) -> str` — the live UI's tool-affordance string (label, plus `" · " + detail` when `describe_args` is non-empty).

- [ ] **Step 1: Write/extend the failing tests**

Add a focused unit test for `_tool_affordance`, and tighten the existing streaming-sink test (`test_on_tool_sink_streams_and_reports_each_tool_call_by_label`, currently at `tests/test_agent_cascade_brain.py:241`) to pass a non-empty `args` so the composed detail is asserted end-to-end.

New test (place next to `test_tool_label_maps_web_search_and_falls_back_for_others`, ~line 257):

```python
def test_tool_affordance_appends_the_identifying_arg():
    # The web-search query and a generic tool's identifying arg are appended after a middle dot;
    # an argument-less call degrades to the bare label (no trailing separator).
    assert (
        brain._tool_affordance(brain.WEB_SEARCH_TOOL_NAME, {"query": "ai house Seattle"})
        == "Searching the web · ai house Seattle"
    )
    assert brain._tool_affordance("read_file", {"path": "notes.md"}) == "Using read_file · notes.md"
    assert brain._tool_affordance("get_time", {}) == "Using get_time"
```

Edit the existing sink test so the scripted call carries a query and the asserted label includes the detail (this is the change that kills the mutation-gate mutant on the reworked `on_tool(...)` line):

```python
def test_on_tool_sink_streams_and_reports_each_tool_call_by_label():
    # The on_tool sink receives the composed affordance (label · identifying arg) for each call.
    labels: list[str] = []
    model = _scripted_model(
        content="",
        tool_calls=[{"name": brain.WEB_SEARCH_TOOL_NAME, "args": {"query": "today's news"}, "id": "c1"}],
    )
    completer = _completer_for(model)
    reply = completer([{"role": "user", "content": "news?"}], on_tool=labels.append)
    assert labels == ["Searching the web · today's news"]
```

> Note: keep the rest of that test body (the `_scripted_model`/`_completer_for` helpers and the `reply` assertion, if any) exactly as it already is — only the `args` value and the `labels ==` expectation change. Read the current body at `tests/test_agent_cascade_brain.py:241` before editing so no surrounding line is lost.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py::test_tool_affordance_appends_the_identifying_arg tests/test_agent_cascade_brain.py::test_on_tool_sink_streams_and_reports_each_tool_call_by_label -v`
Expected: FAIL — `AttributeError: module 'aai_cli.agent_cascade.brain' has no attribute '_tool_affordance'`, and the sink test fails on `['Searching the web'] != ['Searching the web · today's news']`.

- [ ] **Step 3: Implement in `brain.py`**

Add `Mapping` to the `collections.abc` import (line 21):

```python
from collections.abc import Callable, Mapping, Sequence
```

Add the `describe_args` import alongside the other `code_agent` imports (after line 26):

```python
from aai_cli.code_agent.summarize import describe_args
```

Add `_tool_affordance` immediately after `_tool_label` (after line 52):

```python
def _tool_affordance(name: str, args: Mapping[str, object]) -> str:
    """The live UI's tool-affordance string: the label plus its identifying arg.

    Joins the friendly present-tense label (``Searching the web`` / ``Using read_file``) with
    the one identifying argument :func:`describe_args` picks out (a query, path, or URL), so a
    paused turn reads as ``Searching the web · ai house Seattle`` rather than a bare verb. Falls
    back to the bare label when the call carries no arguments.
    """
    label = _tool_label(name)
    detail = describe_args(args)
    return f"{label} · {detail}" if detail else label
```

Change the `on_tool` call inside `_surface_event` (line 285) from:

```python
        on_tool(_tool_label(event.name))
```

to:

```python
        on_tool(_tool_affordance(event.name, event.args))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py -v`
Expected: PASS (the two edited/added tests, plus the rest of the file unchanged).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "assembly live: show a tool call's identifying arg in its affordance

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Spaced `ToolAffordance` widget for live tool calls (`messages.py` + `tui.py`)

**Files:**
- Modify: `aai_cli/code_agent/messages.py` (add `ToolAffordance` after the `Note` class, ~line 31)
- Modify: `aai_cli/agent_cascade/tui.py` (import `ToolAffordance` at line 26; add two CSS rules in the `CSS` block ~line 106–107; rework `show_tool_call` at line 219–226)
- Test: `tests/test_live_tui.py`

**Interfaces:**
- Consumes: `_DIM = "#8a8f98"` and `rich.text.Text` (both already in `messages.py`); the `#log` `VerticalScroll` container and its `.children` (the splash `Static` is always `children[0]`, mounted in `on_mount`); `self._mount(widget)` (`tui.py:300`, mounts into `#log`).
- Produces: `messages.ToolAffordance(text: str, *, tight: bool)` — a dim one-line transcript widget; `tight=True` adds the `-tight` CSS class so `LiveAgentApp` drops its top margin.

- [ ] **Step 1: Write the failing pilot tests**

Replace the existing `test_show_tool_call_mounts_an_inline_affordance` (`tests/test_live_tui.py:141`) with a version that asserts the composed text *and* the block-vs-tight spacing, and update the worker-leg assertion (`tests/test_live_tui.py:339`) to query the new widget. Add `ToolAffordance` to the `messages` import at the top of the test file (line 22).

Import line (line 22) becomes:

```python
from aai_cli.code_agent.messages import (
    AssistantMessage,
    ErrorMessage,
    Note,
    ToolAffordance,
    UserMessage,
)
```

Replace `test_show_tool_call_mounts_an_inline_affordance`:

```python
def test_show_tool_call_mounts_a_spaced_affordance() -> None:
    # A tool call mounts a dim ToolAffordance carrying the composed label; the first call of a
    # turn keeps its top margin (lifts the block off the prompt) and a consecutive call adds the
    # `-tight` class so the two lines don't sprawl.
    async def go() -> None:
        app = _app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.show_tool_call("Searching the web · Boston weather")
            app.show_tool_call("Using read_file · notes.md")
            lines = list(app.query(ToolAffordance))
            assert len(lines) == 2
            assert "Searching the web · Boston weather" in str(lines[0].render())
            assert lines[0].has_class("-tight") is False  # first of the turn -> margin lifts it
            assert lines[1].has_class("-tight") is True  # consecutive -> tight, no extra gap

    _run(go())
```

Update the worker-leg assertion at line 339 (inside `test_worker_drives_the_renderer_and_unmount_closes_audio`) from:

```python
            assert any("Searching the web" in str(n.render()) for n in app.query(Note))
```

to:

```python
            assert any("Searching the web" in str(t.render()) for t in app.query(ToolAffordance))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_live_tui.py::test_show_tool_call_mounts_a_spaced_affordance -v`
Expected: FAIL — `ImportError: cannot import name 'ToolAffordance' from 'aai_cli.code_agent.messages'`.

- [ ] **Step 3: Add the `ToolAffordance` widget in `messages.py`**

Insert after the `Note` class (after line 30):

```python
class ToolAffordance(Static):
    """A dim live tool-call line: the friendly label plus its identifying arg.

    ``Searching the web · ai house Seattle…``. Distinct from :class:`ToolCallLine` (the coding
    agent's ``→ name(args)`` form) — this is the voice TUI's progress affordance, spaced by
    ``LiveAgentApp``: ``tight`` adds the ``-tight`` class so a consecutive call drops the top
    margin the first call of a turn keeps.
    """

    def __init__(self, text: str, *, tight: bool) -> None:
        super().__init__(Text(text, style=_DIM), classes="-tight" if tight else None)
```

- [ ] **Step 4: Wire it into `tui.py`**

Add `ToolAffordance` to the `messages` import (line 26):

```python
from aai_cli.code_agent.messages import (
    AssistantMessage,
    ErrorMessage,
    Note,
    ToolAffordance,
    UserMessage,
)
```

Add two CSS rules at the end of the `CSS` block, just after the `AssistantMessage` rule (after line 107). NOTE the doubled braces — `CSS` is an f-string:

```python
    /* First tool line of a turn keeps a top margin (lifts the block off the prompt); a
       consecutive call adds `-tight` to drop it, so a multi-tool turn stays compact. */
    ToolAffordance {{ margin-top: 1; }}
    ToolAffordance.-tight {{ margin-top: 0; }}
```

Rework `show_tool_call` (lines 219–226). The splash `Static` is mounted before any turn, so `#log` always has at least one child — the last child is a `ToolAffordance` only when the previous mount was itself a tool call:

```python
    def show_tool_call(self, label: str) -> None:
        """Surface the agent's tool use inline as it happens (the live tool affordance).

        A spoken turn that pauses to use a tool would otherwise sit silent on "thinking…"; this
        drops a dim "Searching the web · …" line so the wait reads as progress, not a hang. The
        first such line of a turn is spaced off the prompt; a consecutive call mounts tight.
        """
        log = self.query_one("#log", VerticalScroll)
        tight = isinstance(log.children[-1], ToolAffordance)
        self._mount(ToolAffordance(f"{label}…", tight=tight))
        self._scroll_end()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_live_tui.py -v`
Expected: PASS (the new spacing test, the updated worker-leg test, and the rest of the file).

- [ ] **Step 6: Commit**

```bash
git add aai_cli/code_agent/messages.py aai_cli/agent_cascade/tui.py tests/test_live_tui.py
AAI_ALLOW_COMMIT=1 git commit -m "assembly live: space tool-call lines with a dedicated ToolAffordance widget

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Regenerate the live tool-call visual snapshot

**Files:**
- Modify: `tests/test_tui_snapshots.py` (`test_live_tool_call_note`, line 320–330)
- Modify (regenerate): `tests/__snapshots__/test_tui_snapshots/test_live_tool_call_note.raw`

**Interfaces:**
- Consumes: `LiveAgentApp.show_tool_call(label)` (now mounts `ToolAffordance`); `h.build_live_app()`, `h.freeze_animation`, `h.TERMINAL_SIZE` (existing snapshot harness).

- [ ] **Step 1: Update the snapshot test body to exercise the detail + spacing**

Keep the test name `test_live_tool_call_note` (so the golden filename is regenerated in place, no orphaned `.raw`). Replace its body/docstring (lines 320–330) with two composed-label calls so the SVG pins both the detail and the gap-before-block / tight layout:

```python
def test_live_tool_call_note(snap_compare) -> None:
    """Tool calls mid-turn show the friendly label plus its identifying detail; the block is
    lifted off the prompt by a blank line, and a consecutive call stays tight beneath it."""

    async def run_before(pilot: Pilot[None]) -> None:
        app = pilot.app
        assert isinstance(app, LiveAgentApp)
        h.freeze_animation(app)
        app.show_user_final("what's the weather like in Boston?")
        app.show_tool_call("Searching the web · Boston weather")
        app.show_tool_call("Using read_file · forecast.md")

    assert snap_compare(h.build_live_app(), terminal_size=h.TERMINAL_SIZE, run_before=run_before)
```

- [ ] **Step 2: Confirm it fails against the stale golden**

Run: `uv run pytest tests/test_tui_snapshots.py::test_live_tool_call_note -v`
Expected: FAIL — the painted frame no longer matches the committed `.raw` (new widget text + spacing).

- [ ] **Step 3: Regenerate the golden**

Run: `uv run pytest tests/test_tui_snapshots.py::test_live_tool_call_note --snapshot-update`
Expected: the `.raw` golden is rewritten; pytest reports the snapshot updated.

- [ ] **Step 4: Eyeball the regenerated SVG**

Open `tests/__snapshots__/test_tui_snapshots/test_live_tool_call_note.raw` and confirm, by grouping `<text>` elements by their `y` coordinate (no SVG viewer needed in a headless session):
- both composed labels render (`Searching the web · Boston weather…`, `Using read_file · forecast.md…`),
- there is one blank row between the `» …` user line and the first tool line,
- the two tool lines are on adjacent rows (no blank row between them).

Then re-run to confirm green: `uv run pytest tests/test_tui_snapshots.py::test_live_tool_call_note -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_tui_snapshots.py tests/__snapshots__/test_tui_snapshots/test_live_tool_call_note.raw
AAI_ALLOW_COMMIT=1 git commit -m "assembly live: bless the spaced tool-call visual snapshot

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Full gate + final landing

**Files:** none (verification + a gated final commit if the gate fixes anything).

- [ ] **Step 1: Run the authoritative gate**

Run: `./scripts/check.sh`
Expected: it prints `All checks passed.` Pay attention to the diff-scoped tails — patch coverage (100% on changed lines) and the mutation gate (the `tight` branch in `show_tool_call`, the `if detail else` branch in `_tool_affordance`, and the `· `-separated label must all be killed by the Task 1/2 assertions). The per-surface Textual coverage floor (≥90%) covers `tui.py`/`messages.py`.

- [ ] **Step 2: If the gate flags anything, fix it and re-run**

Address only this feature's findings (e.g. a surviving mutant → add the missing behavioral assertion to the relevant test from Task 1/2). Re-run `./scripts/check.sh` until it prints `All checks passed.` A clean gate records `.git/aai-gate-pass` for the current tree.

- [ ] **Step 3: Final commit (gated) and push**

If Step 2 changed files, commit them normally now (the gate marker matches, so no `AAI_ALLOW_COMMIT` needed):

```bash
git add -- aai_cli/agent_cascade/brain.py aai_cli/code_agent/messages.py aai_cli/agent_cascade/tui.py tests/test_agent_cascade_brain.py tests/test_live_tui.py tests/test_tui_snapshots.py tests/__snapshots__/test_tui_snapshots/test_live_tool_call_note.raw
git commit -m "assembly live: tidy tool-call UX after gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

Then push the branch and open a PR (let it land through the merge queue, per the repo convention):

```bash
git push -u origin live-tool-call-ux
gh pr create --fill
```

---

## Self-Review

**Spec coverage:**
- "Detail — compose label with identifying arg" → Task 1 (`_tool_affordance`, wired into `_surface_event`). ✓
- "Spacing — gap before the block" → Task 2 (`ToolAffordance` + `-tight` class + CSS + `show_tool_call`). ✓
- "`Renderer.tool_call` unchanged; `AgentRenderer` benefits free" → Task 1 changes only the composed string; protocol untouched. ✓
- "Trailing `…` from renderers" → preserved (`f"{label}…"` in `show_tool_call`; `_tool_affordance` never adds it). ✓
- Testing (unit / pilot / visual) → Tasks 1, 2, 3 respectively; behavioral spacing assertions for the mutation gate. ✓
- Out-of-scope guards (no results/spinner, `_tool_label` untouched, in-flight work left alone) → Global Constraints + only-this-feature `git add`. ✓

**Placeholder scan:** No TBD/TODO; every code step shows the literal code; commands have expected output. ✓ (One deliberate "read the current body before editing" note in Task 1 Step 1, because that test's helpers aren't reproduced here — the edit is scoped to two lines.)

**Type consistency:** `_tool_affordance(name: str, args: Mapping[str, object]) -> str` is defined in Task 1 and consumed at `brain.py:285`; `ToolAffordance(text: str, *, tight: bool)` is defined in Task 2's `messages.py` and used identically in `tui.py` and `tests/test_live_tui.py`. The `-tight` class name is consistent across the widget, the CSS, and the pilot assertions. ✓
