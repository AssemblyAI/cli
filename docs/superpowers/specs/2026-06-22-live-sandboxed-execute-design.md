# Sandboxed cowork `execute` + durable memory for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Turn the `assembly live` voice agent (the `agent-cascade` command) from a
read-only assistant into one that can **cowork on the project in your current
directory** — write/edit files, then actually run the project's tools
(`pytest`, `git diff`, `npm run build`) against those edits — and **pick up
where it left off across sessions**. Two capabilities:

1. **Sandboxed, gated `execute`.** Light up deepagents' built-in `execute` tool
   (today bound but inert, because `--files` uses a plain `FilesystemBackend`
   that is not a `SandboxBackendProtocol`). `execute` runs commands **in the
   real cwd**, kernel-confined by an OS sandbox so they can't escape the
   directory or reach the network, and every run is **approved with a TUI
   y/n**.
2. **Durable cross-session memory.** Use deepagents' built-in `MemoryMiddleware`
   to load and persist a per-project memory file, so the agent resumes knowing
   what it was working on.

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Tools are normally auto-approved — a
low-latency spoken turn can't pause for a keyboard confirmation — but `--files`
is the exception: it swaps the in-memory backend for a real-cwd
`FilesystemBackend(virtual_mode=True)` and gates `write_file`/`edit_file` behind
a TUI `y/a/n` approval (`brain._stream_gated` + `agent_cascade.modals`,
resumed via an `InMemorySaver` checkpointer). This work extends that exact
machinery to `execute` and adds a backend that can actually run code.

deepagents adds the `execute` tool automatically when the backend implements
`SandboxBackendProtocol`; for non-sandbox backends it returns an error
("inert"). The shipped backends are `LocalShellBackend` (unrestricted host
shell — deepagents explicitly warns against untrusted use) or a `BaseSandbox`
subclass. `risk.py` already carries shell-risk scoring for `execute` (dormant
today because `execute` isn't gated; this work makes it live).

There is **no first-class Python macOS-sandbox library**. The idiomatic
mechanism is `sandbox-exec -p '<SBPL profile>'` (Apple Seatbelt — still shipping,
used by AI coding-agent sandboxes); on Linux it's the `bwrap` (bubblewrap)
binary. Both are pure-subprocess — no new dependency — which fits this repo
(`S603/S607` are ignored project-wide for controlled shell-outs).

**Prior art — `@anthropic-ai/sandbox-runtime` (srt).** Anthropic's own sandbox
(behind Claude Code's `/sandbox`) uses these same primitives. We borrow its
**posture** (default-allow reads, deny secrets; deny-by-default writes; confine
to the working directory; block network) but **not the dependency** — srt is
Node/TypeScript with no Python binding, so depending on it would add a Node +
`npx` runtime requirement that cuts against the agent's keyless/no-setup ethos.

**Persistence reality.** Core langgraph (already installed) ships only
*in-memory* savers/stores (`InMemorySaver`, `InMemoryStore`); neither persists
to disk. A persistent checkpointer needs `langgraph-checkpoint-sqlite`, which
this repo **deliberately removed** (`e585f08`). deepagents' built-in
`MemoryMiddleware` gives cross-session continuity with **no new dependency** by
loading/persisting an on-disk memory file — the right fit now that cowork has a
real filesystem.

## Decisions

1. **Isolation:** OS-level sandbox. `sandbox-exec -p '<SBPL>'` on macOS, `bwrap`
   on Linux. **Inert (safe refusal) on every other platform or when the sandbox
   binary is missing — never a fallback to unconfined execution.** No new
   dependency.
2. **Scope:** general shell — deepagents' native `execute(command)`.
3. **Activation:** folded into the existing `--files` flag (no new flag).
4. **Workspace — cwd-scoped cowork.** `execute` runs **in the real cwd**.
   Read posture (cribbed from srt): **reads allowed by default** (system + cwd +
   `$HOME`) so tools work, with a **secrets denylist** blocked
   (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.netrc`, `~/.npmrc`, `.env`/`.env.*`,
   `.claude/`). **Writes allowed only within cwd** (plus the OS temp dir), with
   **code-execution-persistence paths write-denied even inside cwd**
   (`.git/hooks/`, shell rc files). **No network.** Cannot escape cwd. Damage is
   bounded to the project directory and git-recoverable.
5. **Gating — `execute` requires y/n.** `execute` joins `write_file`/`edit_file`
   in the `interrupt_on` set and is approved through the existing TUI approver
   (`risk.py`'s shell-risk warning now surfaces on that prompt). The OS sandbox
   is **defense-in-depth**: even an approved command can't reach the network or
   escape cwd.
6. **Persistence — deepagents `MemoryMiddleware`.** When `--files` is on, attach
   `MemoryMiddleware` reading a per-project memory file (`./.deepagents/AGENTS.md`)
   through the cwd backend. The agent maintains it during work; it reloads next
   session. No new dependency. This is *durable working memory*, distinct from
   the in-session `InMemorySaver` (which still exists only to drive
   interrupt/resume within a session).

### Why these, over the alternatives (rejected)

- **Ephemeral scratch dir / fully isolated from cwd** — rejected: that is "run
  arbitrary code safely," not cowork. Confining writes to `/tmp` and deny-reading
  cwd means `execute` can't `pytest` the repo or build the files the agent just
  edited. Cowork requires operating on the real project.
- **`execute` unprompted (trust the sandbox alone)** — rejected: even confined
  to cwd, an approved-by-default agent could delete or rewrite project files; a
  y/n keeps the human in the loop, with the sandbox limiting blast radius.
- **Enumerate-the-allowed-system-read-paths (deny reads by default)** —
  rejected: hand-maintaining the `/usr` / `/System` / `/Library` set a Python
  install needs is the most fragile part of a macOS sandbox. srt is
  default-allow-reads + deny-secrets for exactly this reason; we adopt that.
- **Persistent sqlite checkpointer for cross-session resume** — rejected: it
  re-adds the deliberately-removed `langgraph-checkpoint-sqlite` dep, and the
  deliberate fresh-`thread_id`-per-turn design (avoids re-accumulating history)
  means thread-state resume doesn't map cleanly. `MemoryMiddleware` fits better.
- **`LocalShellBackend` unconfined** — rejected: deepagents itself warns it
  gives no isolation; the sandbox is the whole point.
- **Docker / container sandbox** — rejected: heavy daemon dependency, slow
  per-session cold start for a keyless CLI voice agent.
- **Depend on `@anthropic-ai/sandbox-runtime` directly** — rejected: Node-only,
  adds an `npx` runtime dependency. We borrow its posture, not its code.

## Scope

- **Live-only.** All new code lives in `aai_cli/agent_cascade/`; gated behind
  `--files`. Nothing else in the CLI changes.
- **No new dependency.** Pure subprocess over OS binaries; `MemoryMiddleware`
  and `InMemorySaver` are already available.
- **Speakable contract preserved.** `execute` never raises into the graph; on
  any failure it returns a short string for the agent to speak.

### Out of scope (YAGNI)

- Windows sandboxing → `execute` stays inert there.
- Network access or package installation inside the sandbox.
- Docker / remote / cloud sandboxes.
- Full-transcript checkpointer resume; global/cross-project memory (memory is
  per-project, in cwd).
- A separate `--sandbox`/`--exec` flag or per-tool opt-outs.

## Architecture

### New module: `aai_cli/agent_cascade/sandbox.py`

The entire sandbox concern in one focused, independently-testable module.

- **`class SandboxedShellBackend(LocalShellBackend)`** — inherits
  `FilesystemBackend` file operations rooted at cwd (so
  `read_file`/`write_file`/`edit_file`/`ls`/`glob`/`grep` behave exactly as
  `--files` today) and **overrides `execute()`** so it never delegates to the
  inherited host-shell `execute`. Implementing `SandboxBackendProtocol` (via
  `LocalShellBackend`) is what makes deepagents auto-add the `execute` tool.
  - `execute(command, *, timeout=None) -> ExecuteResponse`: resolve capability →
    render the cwd-scoped policy → run the wrapped command through the injected
    `Runner` with `cwd=<real cwd>` → return combined stdout + exit code. Bounded
    by `timeout` (default + a hard max).
  - **Invariant:** the override must never call `super().execute()` (the
    unconfined host shell). Capability `none` → return a refusal, run nothing.

- **The secrets / persistence denylists (shared constants):** one read-deny
  tuple (credential stores + `.env` + `.claude/`) and one within-cwd write-deny
  tuple (`.git/hooks/`, shell rc files), cribbed from srt's auto-protected set.
  Both renderers consume the same constants so the platforms stay in lockstep
  (a parity test asserts it).

- **Policy rendering (pure functions — the security core):**
  - `render_seatbelt_profile(cwd, tmp, *, read_deny, write_deny) -> str` — SBPL
    with **default-allow reads**: `(version 1)`, `(deny default)`,
    `(allow process-exec*)`, `(allow file-read*)`, then
    `(deny file-read* (subpath …)/(regex …))` per read-deny entry (Seatbelt glob
    patterns handle `.env*`), `(allow file-write* (subpath "<cwd>") (subpath
    "<tmp>"))`, then `(deny file-write* (subpath "<cwd>/.git/hooks") …)` per
    write-deny entry (last-match-wins, so denies override). Network stays denied
    by `(deny default)`.
  - `build_bwrap_argv(cwd, tmp, command, *, read_deny, write_deny) -> list[str]`
    — `bwrap --unshare-all --die-with-parent`, `--ro-bind / /` (whole FS
    read-only = default-allow-reads), `--bind <cwd> <cwd>` (rw) and
    `--bind <tmp> <tmp>`, then `--tmpfs`/`--ro-bind /dev/null` masks over each
    secret path and `--ro-bind` over `.git/hooks` to block writes, network
    unshared. **Platform note:** bubblewrap is path-based, so in-cwd secret-file
    protection (e.g. arbitrary `.env`) is coarser than Seatbelt's glob denies —
    documented as a known asymmetry; the directory-level credential stores
    (`~/.ssh`, …) are masked precisely on both.
  - **Optional hardening:** wrap the inner command with `ulimit -t`/`ulimit -v`
    (CPU/address-space caps) so a runaway can't peg the box inside the timeout.
    Mark the literal caps `# pragma: no mutate` (tuning knobs).

- **Capability detection (injectable):** resolve `"seatbelt" | "bwrap" | "none"`
  from platform + a `which`-style probe. `"none"` → `execute` returns *"I can't
  run code on this system."* and **never** shells out. This
  refuse-don't-fall-back branch is the single most safety-critical line.

- **Seams for hermetic tests:** `Runner = Callable[[list[str], str, int],
  CompletedProcessLike]` (default wraps `subprocess.run` with combined output,
  `cwd`, `timeout`, minimal env) and the capability probe — both injectable so
  the suite asserts *what argv/profile we'd run* with no real sandbox (CI
  reliably has neither binary).

### Edits to `brain.py` (the one shared file, minimal + additive)

- `_build_fs_backend()` returns `SandboxedShellBackend(root_dir=str(Path.cwd()),
  virtual_mode=True)` instead of `FilesystemBackend`. `--files`-off path
  unchanged.
- `_WRITE_TOOLS` becomes `("write_file", "edit_file", "execute")` so `execute`
  is added to `interrupt_on` and flows through the existing approval/resume loop
  (`_stream_gated`/`_decide`). The `InMemorySaver` checkpointer is unchanged
  (still required for in-session interrupt/resume).
- `_graph_kwargs` additionally attaches `MemoryMiddleware(backend=<the
  SandboxedShellBackend>, sources=["./.deepagents/AGENTS.md"])` via
  `create_deep_agent`'s `middleware=` param (confirmed present alongside
  `backend`/`interrupt_on`/`checkpointer`) when `config.files` is on. The
  middleware reads through the cwd backend; the agent updates the file via
  `write_file` (which prompts, like any cwd write).
- `_TOOL_LABELS["execute"] = "Running code"` — the live-UI affordance.
- The system-prompt capability phrasing advertises *"run code to solve problems
  and operate on this project"* only when `execute` is in the bound toolset.

## Boundary / housekeeping

- `subprocess` is fenced by ruff `TID251`; `sandbox.py` gets a deliberate,
  reviewable per-module allowlist entry. The child env is built minimally via
  `core/env.child_env`.
- `risk.py`'s `execute` branch becomes **live** (the shell-risk warning now
  shows on the y/n prompt) — no longer dormant, so its tests assert real
  behavior.
- Stale comments to fix: the "always-bound `execute` … inert" notes in
  `brain.py`; the `--files` paragraph in `aai_cli/CLAUDE.md` (now: sandboxed
  gated code execution + durable memory); the `--files` help string (regenerate
  the affected `--help` snapshot; never hand-edit `.ambr`).
- The memory file lives at `./.deepagents/AGENTS.md` (deepagents convention).
  No new env var / command ⇒ docs-consistency gate stays green; update
  REFERENCE.md/README only if their `--files` description needs it.

## Error handling (cross-cutting)

`execute` is best-effort and never raises into the graph:

- capability `none` → *"I can't run code on this system."*
- sandbox launch failure (`Runner` raises) → a short apology string.
- timeout / non-zero exit → returned as combined output + `exit_code` for the
  model to read aloud (a failed run is information, not an error path).
- user declines the y/n → the standard `_DECLINED` message, same as a declined
  write today.

This mirrors the never-raise contract every live tool follows.

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks. One
`tests/test_agent_cascade_sandbox.py`, fully hermetic via the injected `Runner`
and capability seams — no real sandbox, no sockets.

- **Policy renderers:** `render_seatbelt_profile` asserts `(deny default)` +
  `(allow file-read*)` (default-allow reads), each read-deny path emits a
  `file-read*` deny, **cwd is a `file-write*` subpath**, each write-deny path
  (incl. `.git/hooks`) emits a `file-write*` deny, and no network allow exists;
  `build_bwrap_argv` asserts `--unshare-all`, `--ro-bind / /`, the cwd rw bind,
  the secret masks, and the `.git/hooks` read-only bind. A **parity test**
  asserts both renderers cover the same denylist constants. Mutating any
  allow/deny token, or dropping a denylist entry, must fail a test.
- **`execute()` happy path:** a fake `Runner` asserts the command is wrapped in
  `sandbox-exec -p <profile>` / `bwrap …` with `cwd=<real cwd>`; timeout
  passthrough; output/exit shaping into `ExecuteResponse`.
- **Capability `none`:** asserts the refusal string **and that the `Runner` is
  never invoked** — kills the "fall back to host shell" mutant.
- **Failure modes:** `Runner` raising → apology; non-zero exit → output+exit
  surfaced.
- **brain wiring:** `_build_fs_backend()` returns a `SandboxBackendProtocol`
  backend (so `execute` binds); `execute` **is** in the `--files` `interrupt_on`
  map (so it prompts) and a declined `execute` yields `_DECLINED`;
  `_tool_label("execute")` returns the new label; the capability phrase appears
  when `execute` is bound; `MemoryMiddleware` is attached with the per-project
  source when `--files` is on. Assert exact behavior/strings, not mere
  execution.
- **`risk.py`:** the now-live `execute` branch asserts the dangerous-shell
  warning fires for a destructive command and is `None` for a benign one.

## PR sequence

**Single feature PR.** No new dependency, so no separate `uv.lock` PR. The
change is `sandbox.py` + the `brain.py` wiring (backend, `execute` gating,
`MemoryMiddleware`) + comment/help/doc updates + the tests.
