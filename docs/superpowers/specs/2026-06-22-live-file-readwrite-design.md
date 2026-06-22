# `assembly live` — file read/write in the launch directory

**Date:** 2026-06-22
**Status:** Design approved, pending spec review

## Summary

Give the `assembly live` voice agent the ability to **read and write files in the
directory it is launched in**, opt-in and behind a confirmation gate for writes.
The capability reuses the filesystem plumbing already proven in `assembly code`
(deepagents' filesystem backend + the interrupt/resume approval loop), so `live`
gains files — not a shell — with minimal new surface area.

## Motivation

`assembly live` is the client-orchestrated voice agent (`agent_cascade`): Streaming
STT → a deepagents brain on the LLM Gateway → streaming TTS. Today its toolset is
deliberately tiny and **read-only** (Firecrawl web search when keyed, plus opt-in
read-only MCP tools), because a spoken turn cannot pause for a keyboard
confirmation. Users want the agent to act on local files during a conversation
("read me notes.txt", "save that summary to summary.md") without leaving the voice
session.

## Decisions (locked during brainstorming)

1. **Opt-in, not default.** A new flag enables the capability; default behavior is
   unchanged (tool-free / web-search-only). Mirrors the strictly-opt-in posture of
   `--mcp-config`.
2. **Reads free, writes confirmed.** Read tools auto-approve; `write_file` /
   `edit_file` require explicit confirmation.
3. **Confirmation is a TUI keypress (y/n).** A pending write pauses the turn and the
   voice TUI shows the target path with a `y/n` prompt. Robust and unambiguous;
   reuses `assembly code`'s interrupt/resume `Approver`. (Spoken yes/no was
   considered and rejected as fragile and a larger change to the turn flow.)
4. **Files, not a shell.** Use `FilesystemBackend` (read/write/edit/ls/glob/grep),
   **not** `LocalShellBackend` — so no `execute` tool is exposed.
5. **Rooted at the launch directory (cwd)**, with `virtual_mode=True` blocking
   traversal escapes — identical containment to `assembly code`.

### Open choices to confirm at spec review

- **Flag name:** proposed `--files` (boolean). Alternatives: `--workdir`,
  `--allow-files`. The root is always cwd for now (no path argument — YAGNI).
- **Read-tool gating:** reads ungated (`read_file` / `ls` / `glob` / `grep`
  auto-approve). Only `write_file` / `edit_file` are confirmed.

## Architecture

### Toolset (reuse from `assembly code`)

`assembly code` builds its graph over
`LocalShellBackend(root_dir=cwd, virtual_mode=True)`, which exposes both filesystem
tools **and** the `execute` shell tool. We instead use
`FilesystemBackend(root_dir=cwd, virtual_mode=True)` from `deepagents.backends`,
which provides `read`/`write`/`edit`/`ls`/`glob`/`grep` and **no** `execute`. Same
`virtual_mode` rooting: the model's `/`-rooted paths map under cwd and traversal
escapes are blocked.

`aai_cli/agent_cascade/brain.py::build_graph` gains the backend when the feature is
enabled. Currently `build_graph` calls `create_deep_agent` with no backend (an
in-memory virtual filesystem); enabling files passes the real `FilesystemBackend`.

### Approval (reuse `assembly code`'s interrupt/resume)

When files are enabled, `build_graph`:

- sets `interrupt_on={"write_file": True, "edit_file": True}` (reads are **not**
  gated), and
- attaches an `InMemorySaver` checkpointer (interrupt/resume requires one) plus a
  stable `thread_id` in the per-invoke config.

The brain's completer (`build_completer` / `_run_graph`) gains an
interrupt-resolution loop modeled on `aai_cli/code_agent/session.py::_resolve_interrupts`:
on a write interrupt it calls an injected `Approver(name, args) -> bool` and resumes
the graph with an approve/reject `Command(resume=...)`, looping until the turn no
longer pauses. The `Approver` type and the resume-decision shape are lifted from the
code agent.

When files are **disabled**, none of this is wired — `build_graph` behaves exactly
as today (no backend, no checkpointer, no interrupt_on).

### Confirmation channel (front-end supplies the `Approver`)

The `Approver` is injected from the front-end through `CascadeDeps`, so the engine
and brain stay testable against plain functions.

- **Voice TUI (`LiveAgentApp`)** — interactive mic, human mode. A pending write
  pauses the reply turn; the TUI surfaces the target path and a `y/n` prompt (a
  small approval line/modal — the TUI already owns the keyboard via its `BINDINGS`).
  The reply worker thread blocks on a `threading.Event` that the UI thread sets on
  keypress, then resumes the graph — the same block-the-worker pattern the code
  agent's TUI approver uses.
- **Plain / headless renderer** — file/URL input, `--json`, `-o text`, or non-TTY
  (where `_should_use_tui` is false). No keyboard channel, so the approver
  **auto-denies** writes (reads still work). The declined write is surfaced inline
  so the turn explains itself rather than silently doing nothing.

### Capability advertisement (system prompt)

`brain.build_system_prompt` / `_tool_capabilities` already tailor the prompt to the
bound tools (so the agent never promises a capability it lacks). When the filesystem
tools are bound, add a phrase like "read and write files in your working directory"
to the capability clause. The existing `_SPOKEN_TAIL` still applies — replies stay
short, spoken, and markdown-free even though the agent can now write files. Tool
labels (`_TOOL_LABELS`, shown as the live "…" affordance) get speakable entries:
"Reading a file", "Writing a file", "Editing a file", "Listing files",
"Searching files".

## Data flow (a write turn, TUI)

1. User speaks → STT finalizes a turn → `CascadeSession.on_turn` starts a reply.
2. The reply worker drives the deepagents graph. The model calls `write_file`.
3. The graph **interrupts** (write is in `interrupt_on`). The completer's resolution
   loop calls the injected `Approver` with `("write_file", {path, content, …})`.
4. The TUI approver hops to the UI thread, shows the path + `y/n`, and blocks the
   worker on an `Event` until a keypress sets approve/reject.
5. The completer resumes the graph with the decision. On approve the file is written
   under cwd; on reject the model is told the user declined (the code agent's
   `_DECLINED` message pattern).
6. The graph finishes; the spoken reply streams out through TTS as usual.

## Error handling

- **Reply timeout vs. human think time.** The reply worker runs the graph under a
  60s wall-clock backstop (`_REPLY_TIMEOUT_SECONDS` in `engine.py`). Time spent
  awaiting human approval must **not** count against that deadline, or a slow
  keypress would cut off the write mid-turn. The design excludes approval-wait time
  from the reply timeout (pause/restart the clock around the approval round-trip, or
  restructure so the approval wait is not under the timed call).
- **Containment.** `virtual_mode=True` rejects paths that escape cwd; such a tool
  call fails inside the graph and is surfaced like any other tool error (the existing
  `brain._run_graph` wraps graph/tool failures as a `CLIError` shown in the
  transcript).
- **Headless writes.** Auto-denied (above) — never a silent no-op.

## Out of scope / minimal touch

- **No shell.** `execute` is never bound; `FilesystemBackend` only.
- **No access outside cwd.** No path argument; root is always the launch directory.
- **Default unchanged.** Without the flag, `live` is exactly as today.
- **`--show-code`.** Verify whether the generated SDK snippet models the brain's
  tools at all. If it does not (it likely renders the STT/LLM/TTS cascade, not the
  deepagents toolset), the flag is reflected minimally or not at all — confirmed
  during implementation.

## Testing

All against fakes — no mic, socket, or real disk-escape.

- **Brain (`tests/test_agent_cascade_*`):**
  - File tools bound **only** when the feature is enabled; absent otherwise.
  - `FilesystemBackend` is constructed rooted at cwd with `virtual_mode=True`.
  - A write interrupt invokes the `Approver`; resume with approve runs the write,
    resume with reject relays the decline and does not write.
  - The system prompt advertises file read/write only when the tools are bound.
- **Engine (`tests/test_agent_cascade_engine.py`):**
  - The `Approver` is threaded through `CascadeDeps` to the completer.
  - The reply timeout excludes approval-wait time.
  - The headless/plain renderer's approver auto-denies writes.
- **TUI (`tests/test_live_tui.py` + snapshots):**
  - Snapshot of the approval prompt (path + `y/n`).
  - A `y` keypress approves and `n` rejects, driving the injected approver `Event`.

## Affected files (anticipated)

- `aai_cli/agent_cascade/brain.py` — backend, interrupt_on, checkpointer, approval
  resolution loop, capability phrase, tool labels.
- `aai_cli/agent_cascade/engine.py` — `Approver` on `CascadeDeps`; timeout vs.
  approval-wait handling.
- `aai_cli/agent_cascade/config.py` — config knob for the enabled flag (+ cwd root).
- `aai_cli/agent_cascade/tui.py` — TUI approval prompt + keypress → `Event` approver.
- `aai_cli/commands/agent_cascade/__init__.py` + `_exec.py` — the new flag → options
  → config; wire the plain renderer's auto-deny approver.
- Tests + `--help` / TUI snapshots regenerated.
