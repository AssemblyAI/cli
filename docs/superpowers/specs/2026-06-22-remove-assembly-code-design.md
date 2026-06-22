# Remove `assembly code`, preserve `assembly live`

**Date:** 2026-06-22
**Status:** Approved design

## Goal

Remove the `assembly code` command and everything used **only** by it, while
keeping `assembly live` (`agent_cascade/`) fully working.

`assembly code` is the `commands/code/` command plus the `code_agent/` feature
slice (24 modules). `assembly live` (`agent_cascade/`) currently borrows 8
modules from `code_agent/`. Per the brainstorm decisions, we **relocate those
shared modules into the `agent_cascade/` slice** (its sole consumer now), then
delete `code_agent/` entirely so no orphaned "code" package survives.

`code_gen/` — the `--show-code` SDK-script generator on
`transcribe`/`stream`/`agent`/`live` — is unrelated and stays untouched.

## Dependency map (why removal isn't a clean `rm`)

`assembly live` (`agent_cascade/`) reaches into `code_agent/` for:

- Standalone, no intra-slice deps: `model.py`, `firecrawl_search.py`,
  `banner.py`, `tui_status.py`
- `messages.py` → `summarize.py`
- `modals.py` (`ApprovalScreen`) → `banner`, `risk`, `summarize`, and
  `voice_ui` (TYPE_CHECKING only)
- `risk.py` → one constant (`FETCH_TOOL_NAME`) from `fetch_tool.py`
- `agent.py` → only the `CompiledAgent` Protocol type (live's `brain.py` builds
  its own deepagents graph via `create_deep_agent`; it does not use
  `code_agent.agent`'s orchestration)

Confirmed orphaned dependency: `langgraph-checkpoint-sqlite` is used **only** by
`code_agent/store.py` (`SqliteSaver`); live uses `InMemorySaver` from langgraph
core. `langchain-mcp-adapters`, `deepagents`, `langgraph`, `langchain-firecrawl`,
`langchain-openai` all remain in use by live and stay.

The `CODE` help panel (`help_panels.CODE`) has `assembly code` as its only
member; `assembly live` lives under the `TRANSCRIPTION` panel.

## Plan

### 1. Relocate the live-shared modules into `agent_cascade/`

Move these 8 files `code_agent/` → `agent_cascade/`:
`model.py`, `firecrawl_search.py`, `banner.py`, `tui_status.py`, `messages.py`,
`summarize.py`, `modals.py`, `risk.py`.

Surgeries so the moved set is self-contained:

- **`CompiledAgent` Protocol** — extract from the deleted `agent.py` into the
  relocated `model.py` (or a small `agent_cascade/types.py`); `brain.py` imports
  it from the new location.
- **`risk.py`** — inline the `FETCH_TOOL_NAME` literal instead of importing it
  from the deleted `fetch_tool.py`.
- **`modals.py`** — drop the `TYPE_CHECKING`-only reference to
  `voice_ui._VoiceIO` (deleted).

Update the live consumers to import from `aai_cli.agent_cascade.*`:
`agent_cascade/brain.py`, `agent_cascade/tui.py`,
`commands/agent_cascade/_exec.py`. Fix the stale `code_agent.fetch_tool` comment
in `agent_cascade/weather_tool.py`.

### 2. Delete the code-only surface

- Command: `aai_cli/commands/code/` (`__init__.py`, `_exec.py`).
- `code_agent/` remainder (after the 8 modules move out): `_config_root`,
  `agent`, `ask_tool`, `cli_tool`, `docs_mcp`, `events`, `fetch_tool`, `memory`,
  `prompt`, `render`, `session`, `skills`, `store`, `tui`, `voice`, `voice_ui`,
  `__init__`. The `code_agent/` package directory is removed entirely.

### 3. Tests

- **Relocate & re-point** the tests covering surviving (moved) modules — rename
  to `test_live_*` / `test_agent_cascade_*` and fix imports:
  `test_code_messages`, `test_code_modals`, `test_code_model`, `test_code_risk`,
  `test_code_summarize`, `test_code_tui_status`. These keep the moved modules
  above the 90% project + 90% Textual-TUI coverage floors.
- **Delete** the code-only tests: `test_code_agent`, `test_code_command`,
  `test_code_session_stream`, `test_code_tui`, `test_code_tui_voice`,
  `test_code_tui_voice_switch`, `test_code_voice`.
- **Keep untouched:** all `test_code_gen*`, `test_agent_cascade_show_code`,
  `test_code_gen_agent_cascade` (these are `--show-code`, unrelated).
- **Snapshots:** regenerate the root `--help` golden with `--snapshot-update`
  (the `CODE` panel disappears); delete the code `--help` golden and any
  code-TUI visual-regression snapshots.

### 4. Config, panel, contracts

- **`help_panels.py`:** remove the `CODE` constant and drop it from
  `PANEL_ORDER`.
- **`pyproject.toml`:** remove the `langgraph-checkpoint-sqlite` dependency and
  run `uv lock`. Update the mypy module-override list (drop
  `code_agent.agent/skills/memory/store`; re-point `code_agent.model` →
  `agent_cascade.model`), the ruff per-file-ignores (drop `docs_mcp`/`session`/
  `tui`/`cli_tool`; re-point the `CompiledAgent` `A002` ignore to its new
  location), and the stale `assembly code` comments.
- **`.importlinter`:** remove `aai_cli.code_agent` from the feature-slice
  independence contract and update the comment. `agent_cascade` is already a
  slice, so the moved modules are covered.

### 5. Docs

- **`README.md`:** delete the `assembly code` table row (the `assembly setup`
  row stays).
- **`aai_cli/AGENTS.md`:** rewrite the `code_agent/` subsystem bullet — fold the
  surviving chrome into the `agent_cascade/` bullet; scrub `code_agent`
  mentions.
- Scrub `assembly code` references in the bundled
  `aai_cli/skills/aai-cli/SKILL.md`.
- Leave historical `docs/superpowers/specs/*` design docs as-is (a record).

## Verification

The full `./scripts/check.sh` must end green — especially:

- `vulture` — prune any now-dead exports inside the moved modules.
- `deptry` — no orphaned dependencies remain.
- `lint-imports` — the import-linter contracts hold after the slice removal.
- docs-consistency gate — no doc references `assembly code` anymore.
- diff-scoped mutation + 100% patch-coverage gates on every changed line.

## Out of scope / notes

- The `.claude/worktrees/live-tool-call-impl/` worktree is a concurrent
  session's copy — untouched.
- The working tree is on branch `live-tool-call-ux` with uncommitted
  `agent_cascade` edits. Since this removal heavily edits `agent_cascade`, the
  uncommitted edits should be committed/stashed first so the removal lands on a
  clean base.
