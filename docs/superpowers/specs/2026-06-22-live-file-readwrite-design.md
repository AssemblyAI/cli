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
   **not** `LocalShellBackend`. deepagents' filesystem middleware *always* binds an
   `execute` tool, but with a non-sandbox backend (`FilesystemBackend`) `execute` is
   **inert** — it returns "provide a backend that implements SandboxBackendProtocol"
   and physically cannot run a shell command. So "files, not a shell" holds: we do
   not use a sandbox backend, we do not advertise `execute` in the system prompt, and
   we do not gate it (an inert tool needs no gate). This matches today's behavior —
   the current live graph already binds an inert `execute`. **Search/`grep` is a
   required capability** and is one of the backend's built-in tools, so it comes for
   free (ungated, like the other reads).
5. **Rooted at the launch directory (cwd)**, with `virtual_mode=True` blocking
   traversal escapes — identical containment to `assembly code`.

### Open choices to confirm at spec review

- **Flag name:** proposed `--files` (boolean). Alternatives: `--workdir`,
  `--allow-files`. The root is always cwd for now (no path argument — YAGNI).
- **Read-tool gating:** reads ungated (`read_file` / `ls` / `glob` / `grep`
  auto-approve — including content search via `grep`). Only `write_file` /
  `edit_file` are confirmed.

## Architecture

### Toolset — what actually changes

A key fact discovered during design: `create_deep_agent` **always** installs
deepagents' filesystem middleware, so the **current** live graph already binds
`ls`/`read_file`/`write_file`/`edit_file`/`glob`/`grep` (+ `write_todos`/`task`/inert
`execute`). Today these run against deepagents' default *in-memory* backend, so file
ops touch ephemeral graph state — **not** the launch directory — and the system
prompt never advertises them. They are harmless and unused.

So the feature is **not** "add file tools." It is three focused changes, gated on the
new flag:

1. **Point the backend at the real cwd.** `aai_cli/agent_cascade/brain.py::build_graph`
   passes `FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)` (from
   `deepagents.backends`) instead of relying on the default in-memory backend. Now
   `read_file`/`write_file`/`edit_file`/`grep`/… operate on the launch directory.
   `virtual_mode=True` maps the model's `/`-rooted paths under cwd and blocks
   traversal escapes — identical containment to `assembly code`'s
   `LocalShellBackend`.
2. **Gate writes** (below) — because they now touch real disk.
3. **Advertise the capability** in the system prompt (below).

`execute` stays bound but inert (no sandbox backend); it is neither advertised nor
gated. When the flag is **off**, `build_graph` is unchanged from today (default
in-memory backend, no gating, nothing advertised).

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
  - With the flag on, `build_graph` constructs a real-cwd `FilesystemBackend`
    (`root_dir == str(Path.cwd())`, `virtual_mode=True`); with the flag off it does
    not (default in-memory backend, as today). Assert by injecting/patching the
    backend factory seam rather than introspecting langgraph internals.
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
