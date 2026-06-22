# `assembly live` tool-call UX: richer + better-spaced lines

**Date:** 2026-06-22
**Status:** Approved (design)
**Scope:** `assembly live` (the agent cascade) voice TUI only.

## Problem

In the live voice TUI, an agent's tool calls render as flat dim-gray lines —
`Searching the web…`, `Using read_file…` — that are (a) packed flush against the
user prompt and each other with no breathing room, and (b) missing the one detail
that makes a call legible: *what* it searched for or *which* file it read.

The fix is intentionally light-touch: a bit more detail when it's available, and a
bit more vertical spacing. It does **not** add tool *results*, a spinner, completion
states, or expand the friendly-label map.

## Current behavior

- `brain._surface_event` (`aai_cli/agent_cascade/brain.py:285`) feeds the live UI a
  tool affordance via `on_tool(_tool_label(event.name))` — it has `event.args` in
  hand but discards them.
- `_tool_label` (`brain.py:50`) maps `firecrawl_search → "Searching the web"` and
  falls back to `"Using <name>"` for everything else.
- The TUI's `show_tool_call` (`aai_cli/agent_cascade/tui.py:219`) mounts a shared
  `Note` widget as `Note(f"{label}…")`. `Note` carries no margin, so tool lines pack
  flush together and against the user prompt.
- The non-TUI `AgentRenderer.tool_call` (`aai_cli/agent/render.py`) also just appends
  `…` to the label it's handed.

There is already a shared helper, `describe_args()`
(`aai_cli/code_agent/summarize.py:36`), that extracts the single identifying argument
(`query`/`path`/`url`/`command`/…), clipped to 60 chars — it's what `assembly code`
uses to render `→ write_file(app.py)`. `brain.py` already imports from `code_agent`
(`events`, `firecrawl_search`), so the import direction is established and
import-linter-clean (feature-slice → feature-slice; only `commands` imports are
forbidden).

## Design

Two small, disjoint changes.

### 1. Detail — compose the label with its identifying argument

In `brain.py`, add a helper that joins the friendly label with the identifying arg:

```python
def _tool_affordance(name: str, args: Mapping[str, object]) -> str:
    """The tool label plus its identifying arg: 'Searching the web · ai house Seattle'."""
    label = _tool_label(name)
    detail = describe_args(args)          # reuses code_agent.summarize
    return f"{label} · {detail}" if detail else label
```

- Import `describe_args` from `aai_cli.code_agent.summarize`.
- In `_surface_event`, replace `on_tool(_tool_label(event.name))` with
  `on_tool(_tool_affordance(event.name, event.args))`.
- `_tool_label` is untouched — only `firecrawl_search` stays mapped; everything else
  remains `Using <name>` (so a generic call reads `Using read_file · notes.md`).
- The `Renderer.tool_call(label)` protocol signature is **unchanged** — only the
  string gets richer, so the non-TUI `AgentRenderer` benefits with no edit.
- The trailing `…` keeps being appended by the renderers (not by `_tool_affordance`),
  so it lands after the detail: `Searching the web · ai house Seattle…`.
- When a tool has no args, `describe_args({})` returns `""` and the `if detail` guard
  keeps the bare label.

Examples:

| Tool call | Before | After |
| --- | --- | --- |
| `firecrawl_search(query="ai house Seattle")` | `Searching the web…` | `Searching the web · ai house Seattle…` |
| `read_file(path="notes.md")` | `Using read_file…` | `Using read_file · notes.md…` |
| `some_tool()` (no args) | `Using some_tool…` | `Using some_tool…` |

### 2. Spacing — "gap before the block"

The chosen layout: one blank line above the *first* tool line of a turn (lifting the
block off the prompt), with consecutive tool calls staying tight.

```
» Yeah, the AI house story.

Searching the web · ai house Seattle…
Using read_file · notes.md…
Searching the web · AI2 incubator…

AI House is the new name for Seattle's…
```

- Add a dedicated `ToolLine(Static)` widget in `agent_cascade/tui.py` (dim text, like
  `Note`), so the tool line is CSS-targetable without affecting other `Note` asides.
- Style it at the app-CSS level, mirroring the existing
  `AssistantMessage { margin-top: 1; }` rule:
  - `ToolLine { margin-top: 1; }`
  - `ToolLine.-tight { margin-top: 0; }`
- In `show_tool_call`, inspect the `#log` container's last child: if it is already a
  `ToolLine`, mount the new one with the `-tight` class (no gap); otherwise the default
  top margin applies, separating the block from the prompt.
- `Note` stays in use for the interrupted/cancelled asides.

## Components touched

- `aai_cli/agent_cascade/brain.py` — add `_tool_affordance`, import `describe_args`,
  call it from `_surface_event`.
- `aai_cli/agent_cascade/tui.py` — add `ToolLine` widget + CSS rules; rework
  `show_tool_call` to choose tight vs. spaced and mount `ToolLine` instead of `Note`.

No change to `Renderer`, `engine.py`, `AgentRenderer`, or `_tool_label`.

## Testing

- **Unit** (`tests/test_agent_cascade_brain.py`): `_tool_affordance` appends the query
  for `firecrawl_search`, the identifying arg for a generic tool, and degrades to the
  bare label when args are empty. (Sits beside the existing
  `test_tool_label_maps_web_search_and_falls_back_for_others`.)
- **Pilot** (`tests/test_live_tui.py`): update the existing
  `test_show_tool_call_mounts_an_inline_affordance` (and the worker-leg assertion in
  `test_worker_drives_the_renderer_and_unmount_closes_audio`) to query `ToolLine` and
  assert the composed text. Add a test that the first tool line of a turn lacks
  `-tight` (has the margin) and the second carries `-tight` — the behavioral invariant
  that kills the mutation-gate mutant deterministically, per `tests/CLAUDE.md`.
- **Visual** (`tests/test_tui_snapshots.py`): regenerate the live-TUI golden so the new
  spacing/widget is blessed; eyeball the SVG text before committing.

## Out of scope

- Tool *results*, completion states, spinners, or duration.
- Expanding `_TOOL_LABELS` to map more tool names to friendly verbs.
- The in-flight uncommitted work in the tree (push-to-talk/mute, model swap to
  `kimi-k2.5`, interrupt-behavior changes) — built on top of, left untouched.
