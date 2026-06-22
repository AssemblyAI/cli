# Remove `assembly code` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `assembly code` command and all code used only by it, while keeping `assembly live` (`agent_cascade/`) fully working.

**Architecture:** `assembly code` = `commands/code/` + the `code_agent/` slice (24 modules). `assembly live` borrows 8 of those modules. We **relocate the 8 shared modules into the `agent_cascade/` slice** (its sole remaining consumer), surgically strip the code-only voice path out of `modals.py`, re-point all live imports, then delete `code_agent/` and `commands/code/` entirely. `code_gen/` (`--show-code`) is unrelated and untouched.

**Tech Stack:** Python 3.12–3.13, Typer CLI, `uv`, deepagents/langgraph/langchain, Textual TUI, pytest + syrupy snapshots.

## Global Constraints

- Run every tool through `uv run` (locked env). The authoritative gate is `./scripts/check.sh` — it must print `All checks passed.` before the work is done.
- Commits are gated: a PreToolUse hook blocks `git commit` unless `check.sh` passed for the current tree. For intermediate WIP commits use `AAI_ALLOW_COMMIT=1 git commit …`; run the full gate before the **final** commit of the branch.
- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Help copy is terse, imperative, sentence-case, **no trailing period**. `--help` goldens are syrupy `.ambr` — regenerate with `--snapshot-update`, never hand-edit.
- Errors → stderr, data → stdout. Patch coverage must be 100% vs `origin/main`; the diff-scoped mutation gate requires changed lines to be assertion-covered.
- The `.claude/worktrees/live-tool-call-impl/` directory is another session's worktree — **never touch it**.
- **Dependency change is in-scope here** (the user OK'd folding it in): removing `langgraph-checkpoint-sqlite` rewrites `uv.lock`. Keep all other deps.

---

## Pre-flight (do once before Task 1)

The working tree is on branch `live-tool-call-ux` with uncommitted `agent_cascade` edits, and this removal heavily edits `agent_cascade`. Land the removal on a clean base.

- [ ] **Step 1: Confirm a clean base**

```bash
git -C /Users/alexkroman/Code/docs/assemblyai-cli status --short
```

If `aai_cli/commands/agent_cascade/` or `aai_cli/agent_cascade/` files are dirty, commit or stash them first (coordinate with the user — they own that in-flight work). Do not start Task 1 until `git status --short` shows only files this plan will touch.

- [ ] **Step 2: Sanity-check the current live command works**

```bash
uv run assembly live --help
```

Expected: help text renders, exit 0. This is the smoke test you re-run after every task.

---

## Task 1: Relocate the 8 shared modules into `agent_cascade/`

The 8 modules used by live move out of `code_agent/` into `agent_cascade/`. This is one atomic refactor (partial moves break imports). Move leaf modules first, then dependents.

**Files:**
- Move (`git mv aai_cli/code_agent/X.py aai_cli/agent_cascade/X.py`): `model.py`, `firecrawl_search.py`, `banner.py`, `tui_status.py`, `summarize.py`, `risk.py`, `messages.py`, `modals.py`
- Modify: `aai_cli/agent_cascade/brain.py`, `aai_cli/agent_cascade/tui.py`, `aai_cli/agent_cascade/weather_tool.py`, `aai_cli/commands/agent_cascade/_exec.py`
- Modify (after move): the moved `risk.py`, `messages.py`, `modals.py`, `brain.py` (intra-import re-points + surgeries)

**Interfaces:**
- Produces (new locations, same signatures): `aai_cli.agent_cascade.model.build_model(...)`, `aai_cli.agent_cascade.firecrawl_search.build_web_search_tool()` + `WEB_SEARCH_TOOL_NAME`, `aai_cli.agent_cascade.banner`, `aai_cli.agent_cascade.tui_status`, `aai_cli.agent_cascade.summarize.{describe_args,full_args,summarize_call,summarize_result}`, `aai_cli.agent_cascade.risk`, `aai_cli.agent_cascade.messages.{AssistantMessage,ErrorMessage,Note,UserMessage}`, `aai_cli.agent_cascade.modals.ApprovalScreen`
- Produces: `aai_cli.agent_cascade.brain.CompiledAgent` (Protocol, extracted from the deleted `agent.py`)

- [ ] **Step 1: Move the 6 leaf/standalone modules**

```bash
cd /Users/alexkroman/Code/docs/assemblyai-cli
for m in model firecrawl_search banner tui_status summarize risk; do
  git mv aai_cli/code_agent/$m.py aai_cli/agent_cascade/$m.py
done
```

- [ ] **Step 2: Inline `FETCH_TOOL_NAME` in the moved `risk.py`**

`risk.py` imported `FETCH_TOOL_NAME` from `code_agent/fetch_tool.py` (value `"fetch_url"`), which is being deleted. Replace the import with a module-level literal.

In `aai_cli/agent_cascade/risk.py`, delete the line:

```python
from aai_cli.code_agent.fetch_tool import FETCH_TOOL_NAME
```

and add, near the top of the module body (after the imports):

```python
# The fetch tool's name, inlined here — its defining module lived in the removed
# `assembly code` agent. Risk scoring is purely advisory.
FETCH_TOOL_NAME = "fetch_url"
```

(The existing `elif name == FETCH_TOOL_NAME:` reference now resolves to this local constant.)

- [ ] **Step 3: Move `messages.py` and re-point its `summarize` import**

```bash
git mv aai_cli/code_agent/messages.py aai_cli/agent_cascade/messages.py
```

In `aai_cli/agent_cascade/messages.py`, change:

```python
from aai_cli.code_agent.summarize import summarize_call, summarize_result
```

to:

```python
from aai_cli.agent_cascade.summarize import summarize_call, summarize_result
```

- [ ] **Step 4: Move `modals.py`, re-point imports, and strip the code-only voice path**

```bash
git mv aai_cli/code_agent/modals.py aai_cli/agent_cascade/modals.py
```

In `aai_cli/agent_cascade/modals.py` make these edits:

1. Re-point the surviving imports:

```python
from aai_cli.agent_cascade import banner, risk
from aai_cli.agent_cascade.summarize import describe_args, full_args
```

2. Delete the `TYPE_CHECKING` import of `_VoiceIO`:

```python
    from aai_cli.code_agent.voice_ui import _VoiceIO
```

3. Delete the voice helpers `_spawn(...)` and `approval_from_speech(...)` (lines ~34–55) — they exist only for the spoken-answer path that live never uses.

4. In `ApprovalScreen`: drop the `voice` parameter and `self._voice`/`self._answered` voice bookkeeping, delete the `on_mount` voice branch (`_spawn(lambda: self._drive_by_voice(voice))`), and delete `_drive_by_voice(...)` and `_spoken_prompt(...)`. The keyboard path (`compose`, `action_expand/approve/auto/reject`, `_decide`, `_detail_markup`) stays. Final constructor signature:

```python
    def __init__(self, name: str, args: Mapping[str, object]) -> None:
```

5. Delete the entire `AskScreen` class (lines ~159–end) — live's voice-only TUI never opens an ask modal; it is code-only.

6. Clean the module docstring's voice references and remove now-unused imports (`Callable`, `Input`, `threading`) — the post-edit ruff hook will not auto-remove them, so delete by hand; the gate's `ruff check` would otherwise fail.

- [ ] **Step 5: Extract `CompiledAgent` into `brain.py` and re-point brain's imports**

In `aai_cli/agent_cascade/brain.py`:

1. Replace the import line:

```python
from aai_cli.code_agent.agent import CompiledAgent
```

with the Protocol defined locally (copied verbatim from the deleted `agent.py`), placed after the existing imports:

```python
class CompiledAgent(Protocol):
    """The slice of the compiled langgraph graph the live reply leg drives.

    A structural type so we needn't name langgraph's deeply-generic
    ``CompiledStateGraph`` (and don't drag its type params through our code).
    """

    def invoke(
        self, input: object, config: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Run one step of the graph, returning the updated state (incl. messages)."""
```

Ensure `Protocol` and `Mapping` are imported at the top of `brain.py` (`from typing import Protocol`, `from collections.abc import Mapping`) — add whichever is missing.

2. Re-point the two remaining code_agent imports:

```python
from aai_cli.agent_cascade.firecrawl_search import WEB_SEARCH_TOOL_NAME
```

and inside the functions (lines ~220, ~287):

```python
    from aai_cli.agent_cascade.firecrawl_search import build_web_search_tool
    from aai_cli.agent_cascade.model import build_model
```

- [ ] **Step 6: Re-point `tui.py`, `_exec.py`, and the `weather_tool.py` comment**

In `aai_cli/agent_cascade/tui.py`:

```python
from aai_cli.agent_cascade import banner, tui_status
from aai_cli.agent_cascade.messages import AssistantMessage, ErrorMessage, Note, UserMessage
from aai_cli.agent_cascade.modals import ApprovalScreen
```

In `aai_cli/commands/agent_cascade/_exec.py`:

```python
from aai_cli.agent_cascade import firecrawl_search
```

In `aai_cli/agent_cascade/weather_tool.py`: change the comment mentioning `code_agent.fetch_tool` to reference the behavior generically (no `code_agent` path).

- [ ] **Step 7: Verify no `code_agent` import remains in the live slice**

```bash
grep -rn "code_agent" aai_cli/agent_cascade aai_cli/commands/agent_cascade
```

Expected: **no output**.

- [ ] **Step 8: Verify the live command still imports & runs**

```bash
uv run assembly live --help && uv run python -c "import aai_cli.agent_cascade.brain, aai_cli.agent_cascade.tui, aai_cli.agent_cascade.modals"
```

Expected: help renders (exit 0), import succeeds with no error.

- [ ] **Step 9: Relocate the moved-module tests and re-point live tests**

Rename the tests that cover the moved modules (keep their assertions; drop the now-deleted voice/AskScreen cases in modals):

```bash
git mv tests/test_code_model.py        tests/test_live_model.py
git mv tests/test_code_messages.py     tests/test_live_messages.py
git mv tests/test_code_risk.py         tests/test_live_risk.py
git mv tests/test_code_summarize.py    tests/test_live_summarize.py
git mv tests/test_code_tui_status.py   tests/test_live_tui_status.py
git mv tests/test_code_modals.py       tests/test_live_modals.py
```

In each relocated test, change `from aai_cli.code_agent.X` → `from aai_cli.agent_cascade.X`. In `tests/test_live_modals.py`, **delete** every test that imports/uses `AskScreen`, `approval_from_speech`, or passes `voice=`/`FakeVoice` (those targets no longer exist); keep the keyboard `ApprovalScreen` tests, dropping the `voice=` argument.

Re-point the already-live tests and the snapshot helpers:
- `tests/test_agent_cascade_brain.py`: `from aai_cli.code_agent import model …` / `firecrawl_search` → `from aai_cli.agent_cascade import …`.
- `tests/test_live_tui.py`: re-point every `aai_cli.code_agent.*` import to `aai_cli.agent_cascade.*`.
- `tests/_tui_snapshot.py` and `tests/test_tui_snapshots.py`: re-point `ApprovalScreen` and any moved-module imports to `aai_cli.agent_cascade.*`; **remove** the `AskScreen` import and the AskScreen snapshot case (it is code-only). Leave the `test_live_*` snapshot cases.

- [ ] **Step 10: Run the relocated + live tests**

```bash
uv run pytest tests/test_live_model.py tests/test_live_messages.py tests/test_live_risk.py \
  tests/test_live_summarize.py tests/test_live_tui_status.py tests/test_live_modals.py \
  tests/test_live_tui.py tests/test_agent_cascade_brain.py -q
```

Expected: all pass. Fix import/strip fallout until green.

- [ ] **Step 11: Commit (WIP — gate not yet run)**

```bash
git add -A
AAI_ALLOW_COMMIT=1 git commit -m "refactor(live): relocate shared agent modules from code_agent into agent_cascade"
```

---

## Task 2: Delete the code-only `code_agent/` modules, `commands/code/`, and code-only tests

After Task 1, `code_agent/` holds only code-only modules. Remove them, the command, and the code-only tests.

**Files:**
- Delete dir: `aai_cli/commands/code/`
- Delete dir: `aai_cli/code_agent/` (now contains only: `__init__.py`, `_config_root.py`, `agent.py`, `ask_tool.py`, `cli_tool.py`, `docs_mcp.py`, `events.py`, `fetch_tool.py`, `memory.py`, `prompt.py`, `render.py`, `session.py`, `skills.py`, `store.py`, `tui.py`, `voice.py`, `voice_ui.py`)
- Delete tests: `tests/test_code_agent.py`, `tests/test_code_command.py`, `tests/test_code_session_stream.py`, `tests/test_code_tui.py`, `tests/test_code_tui_voice.py`, `tests/test_code_tui_voice_switch.py`, `tests/test_code_voice.py`
- Delete code-only TUI snapshot rasters: `tests/__snapshots__/test_tui_snapshots/test_code_*.raw`

- [ ] **Step 1: Confirm `code_agent/` has no remaining live consumer**

```bash
cd /Users/alexkroman/Code/docs/assemblyai-cli
grep -rln "code_agent" aai_cli/ | grep -v __pycache__
```

Expected: only `aai_cli/AGENTS.md` (a doc, handled in Task 5). If any `.py` under `aai_cli/` still references `code_agent`, stop and re-point it (Task 1 missed something).

- [ ] **Step 2: Delete the command, the slice, and code-only tests/snapshots**

```bash
git rm -r aai_cli/commands/code aai_cli/code_agent
git rm tests/test_code_agent.py tests/test_code_command.py tests/test_code_session_stream.py \
       tests/test_code_tui.py tests/test_code_tui_voice.py tests/test_code_tui_voice_switch.py \
       tests/test_code_voice.py
git rm tests/__snapshots__/test_tui_snapshots/test_code_*.raw
```

- [ ] **Step 3: Verify command discovery drops `code` and the suite still collects**

```bash
uv run assembly --help | grep -i "coding agent" ; echo "exit: $?"
uv run python -c "from aai_cli import command_registry; names=[c for r in command_registry.discover() for c in r.spec.commands]; assert 'code' not in names, names; assert 'live' in names; print('ok')"
uv run pytest --collect-only -q 2>&1 | tail -5
```

Expected: the `grep` finds nothing (exit 1 from grep is fine — the "Coding Agent" panel is gone); the discovery assertion prints `ok`; collection reports no import errors.

- [ ] **Step 4: Commit (WIP)**

```bash
git add -A
AAI_ALLOW_COMMIT=1 git commit -m "feat(code): remove the assembly code command and its code_agent slice"
```

---

## Task 3: Remove the `CODE` help panel and regenerate snapshots

**Files:**
- Modify: `aai_cli/help_panels.py`
- Modify: `tests/_snapshot_surface.py:29`
- Regenerate: `tests/__snapshots__/test_snapshots_help_root.ambr`, `tests/__snapshots__/test_snapshots_help_run.ambr`

- [ ] **Step 1: Drop the `CODE` panel**

In `aai_cli/help_panels.py`, delete the `CODE = "Coding Agent" …` line and remove `CODE` from the `PANEL_ORDER` tuple.

- [ ] **Step 2: Drop the `CODE` entry from the snapshot partition**

In `tests/_snapshot_surface.py`, delete the line:

```python
    help_panels.CODE: "run",
```

(`code` was the panel's only member; `live` stays mapped via `TRANSCRIPTION: "run"`.)

- [ ] **Step 3: Regenerate the affected `--help` goldens**

```bash
uv run pytest tests/test_snapshots_help_root.py tests/test_snapshots_help_run.py \
  tests/test_snapshots_help_groups.py --snapshot-update -q
```

- [ ] **Step 4: Verify the snapshots and group guard pass cleanly (no update)**

```bash
uv run pytest tests/test_snapshots_help_root.py tests/test_snapshots_help_run.py \
  tests/test_snapshots_help_groups.py -q
```

Expected: pass, with no snapshots reported as updated. Inspect the `git diff` of the `.ambr` files to confirm the only change is the removed `code` command / `Coding Agent` panel.

- [ ] **Step 5: Commit (WIP)**

```bash
git add -A
AAI_ALLOW_COMMIT=1 git commit -m "chore(help): drop the Coding Agent panel after removing assembly code"
```

---

## Task 4: Clean `pyproject.toml`, `.importlinter`, and drop the orphaned dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `.importlinter`
- Modify: `uv.lock` (via `uv lock`)

- [ ] **Step 1: Remove the orphaned dependency**

In `pyproject.toml`, delete the `"langgraph-checkpoint-sqlite>=3.1.0",` dependency line (only the deleted `code_agent/store.py` used `SqliteSaver`; live uses `InMemorySaver` from langgraph core). Leave `deepagents`, `langgraph`, `langchain-mcp-adapters`, `langchain-firecrawl`, `langchain-openai` — live still uses them.

- [ ] **Step 2: Re-lock**

```bash
uv lock
```

- [ ] **Step 3: Update mypy module-overrides and ruff per-file-ignores**

In `pyproject.toml`:
- In the mypy per-module override list, delete `"aai_cli.code_agent.agent"`, `"aai_cli.code_agent.skills"`, `"aai_cli.code_agent.memory"`, `"aai_cli.code_agent.store"` and change `"aai_cli.code_agent.model"` → `"aai_cli.agent_cascade.model"`.
- In `[tool.ruff.lint.per-file-ignores]`, delete the entries for `aai_cli/code_agent/docs_mcp.py`, `aai_cli/code_agent/session.py`, `aai_cli/code_agent/tui.py`, `aai_cli/code_agent/cli_tool.py`, and the `A002` entry for `aai_cli/code_agent/agent.py`. Add `A002` for the `CompiledAgent`-hosting module — append to the existing `aai_cli/agent_cascade/brain.py` ignore (it already appears in the docstring-ignore list): `"aai_cli/agent_cascade/brain.py" = ["A002"]` (merge with any existing key for that file).
- In the docstring-coverage ignore list, remove `"aai_cli/code_agent"` and `"aai_cli/commands/code"` (keep `aai_cli/agent_cascade/brain.py`).
- Update the stale `# assembly code …` comments in the dependency section and the snapshot comment to read `live` / `agent_cascade`.

In `.importlinter`: remove `aai_cli.code_agent` from the feature-slice independence contract's module list and drop it from the explanatory comment (line ~16). `aai_cli.agent_cascade` is already listed.

- [ ] **Step 4: Verify the static-analysis gates pass**

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run lint-imports
uv run deptry .
uv lock --check
```

Expected: all pass; `deptry` reports no obsolete/missing dependency; `uv lock --check` clean.

- [ ] **Step 5: Commit (WIP)**

```bash
git add -A
AAI_ALLOW_COMMIT=1 git commit -m "chore(deps): drop langgraph-checkpoint-sqlite + code_agent lint config"
```

---

## Task 5: Update docs and run the full gate

**Files:**
- Modify: `README.md`
- Modify: `aai_cli/AGENTS.md`
- Verify: `REFERENCE.md`, `aai_cli/skills/aai-cli/SKILL.md` (no `assembly code` refs expected — confirm)

- [ ] **Step 1: Remove the README command-table row**

In `README.md`, delete the `| `assembly code` | …` table row. Leave the `assembly setup` row and the "Agent-ready" bullet (they describe `setup`, not `code`).

- [ ] **Step 2: Rewrite the `aai_cli/AGENTS.md` subsystem docs**

In `aai_cli/AGENTS.md`: delete the `code_agent/` subsystem bullet. In the `agent_cascade/` bullet, replace the phrase that says it reuses `assembly code`'s chrome (`code_agent.banner`/`messages`/`tui_status`/`modals.ApprovalScreen`) with a statement that those modules **now live in `agent_cascade/`**. Remove `code_agent` from the feature-slice list in the layout section.

- [ ] **Step 3: Confirm no stray `assembly code` references remain in shipped docs/skill**

```bash
cd /Users/alexkroman/Code/docs/assemblyai-cli
grep -rn "assembly code\|code_agent" README.md REFERENCE.md aai_cli/skills/aai-cli/SKILL.md aai_cli/AGENTS.md
```

Expected: no output. Fix any hit (the docs-consistency gate fails on a doc-referenced command that no longer exists).

- [ ] **Step 4: Run the full authoritative gate**

```bash
./scripts/check.sh
```

Expected: ends with `All checks passed.` Pay attention to: `vulture` (prune any export inside a moved module that is now dead — e.g. a `summarize`/`risk` helper only the old code TUI used), the docstring-coverage gate, the TUI ≥90% coverage floor, `diff-cover` 100% patch coverage, and the diff-scoped mutation gate. Iterate with targeted commands, then re-run `./scripts/check.sh` to completion.

- [ ] **Step 5: Final commit (gated)**

Because `check.sh` passed, the commit hook is satisfied — no `AAI_ALLOW_COMMIT` needed:

```bash
git add -A
git commit -m "docs: drop assembly code from README and architecture guide"
```

---

## Self-Review notes (for the executor)

- **Coverage of moved code:** the moved modules (`model`, `firecrawl_search`, `banner`, `tui_status`, `summarize`, `risk`, `messages`, `modals`) keep their relocated tests; if stripping `modals` voice/AskScreen drops coverage below the TUI floor, add keyboard-path assertions rather than re-adding dead code.
- **Mutation gate:** the `risk.py` inlined `FETCH_TOOL_NAME` and the `modals` constructor change are changed lines — make sure a relocated test asserts behavior that depends on them (e.g. a risk-scoring test for `fetch_url`, an `ApprovalScreen` keyboard-decision test), or the surviving mutant fails the gate.
- **Don't touch** `code_gen/`, any `test_code_gen*`, `test_agent_cascade_show_code`, or the `.claude/worktrees/` directory.
