# Hands-free sandboxed cowork for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Turn the `assembly live` voice agent (the `agent-cascade` command) into one that
can **cowork on the project in your current directory, hands-free**. Today, even
with `--files`, the agent can read and edit files (each edit gated by a
*keyboard* y/a/n) but it cannot run code, remember anything across sessions,
delegate, or be approved by voice. This work adds four capabilities — all folded
into the existing `--files` flag:

1. **Sandboxed, gated `execute`.** Light up deepagents' built-in `execute` tool
   (today bound but inert, because `--files` uses a plain `FilesystemBackend`
   that is not a `SandboxBackendProtocol`). `execute` runs commands **in the
   real cwd** — so it can `pytest` the repo, `git diff`, `npm run build` the
   files the agent just edited — kernel-confined by an OS sandbox so they can't
   escape the directory or reach the network, and every run is **approved**.
2. **Durable cross-session memory.** Enable deepagents' built-in
   `MemoryMiddleware` (via `create_deep_agent(memory=…)`) over a per-project
   memory file, so the agent resumes knowing what it was working on.
3. **Delegation via the `task` tool.** Wire up deepagents' subagents (available
   but unwired) so the agent can hand a focused multi-step subtask to a
   fresh-context, gateway-bound helper, keeping the main voice turn lean.
4. **Spoken approval.** The approval gate accepts an unambiguous **spoken**
   yes/no — not only a keypress — so the safety gate doesn't contradict the
   hands-free premise (with a keyboard fallback for the highest-risk commands).

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Tools are normally auto-approved — a
low-latency spoken turn can't pause for a confirmation — but `--files` is the
exception: it swaps the in-memory backend for a real-cwd
`FilesystemBackend(virtual_mode=True)` and gates `write_file`/`edit_file` behind
a **keyboard** y/a/n approval (`brain._stream_gated` brackets the wait with
`ApprovalPause` events and calls an injected `Approver`; the voice TUI supplies
it via `agent_cascade.modals.ApprovalScreen`; headless runs auto-deny via
`_exec._deny_writes`; resumed through an `InMemorySaver` checkpointer). This
work extends that exact machinery — a sandbox-capable backend, `execute` in the
gate, voice-aware approval — without replacing it.

deepagents adds the `execute` tool automatically when the backend implements
`SandboxBackendProtocol`; for non-sandbox backends it returns an error
("inert"). The shipped backends are `LocalShellBackend` (unrestricted host
shell — deepagents explicitly warns against untrusted use) or a `BaseSandbox`
subclass. `risk.py` already carries shell-risk scoring for `execute` (dormant
today because `execute` isn't gated; this work makes it live — and reuses it to
pick the highest-risk tier for the keyboard fallback). Subagents are likewise
*available but unwired*: `SubAgentMiddleware` raises "At least one subagent must
be specified" and `create_deep_agent` only adds the `task` node when
`subagents=[…]` is passed; `assembly live` passes none today.

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
   *Reads* allowed by default (system + cwd + `$HOME`) so tools work, with a
   **secrets read-denylist** (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.netrc`,
   `~/.npmrc`, `.env`/`.env.*`, `.claude/`). *Writes* allowed only within cwd
   (plus the OS temp dir), with a **persistence write-denylist** even inside cwd
   (`.git/hooks/`; and shell rc files, which only fall inside the write region
   in the `cwd == $HOME` case but are denied to cover it). **No network.** Can't
   escape cwd. Damage is bounded to the project dir and git-recoverable.
5. **Gating — every mutation is approved.** `execute` joins
   `write_file`/`edit_file` in the `interrupt_on` set and flows through the
   existing approver (`risk.py`'s shell-risk warning surfaces on the prompt). The
   OS sandbox is **defense-in-depth**: even an approved command can't reach the
   network or escape cwd.
6. **Persistence — `MemoryMiddleware` via `memory=`.** When `--files` is on, pass
   `create_deep_agent(memory=["./.deepagents/AGENTS.md"])`; deepagents builds the
   `MemoryMiddleware` over the cwd backend itself. The agent maintains the file
   during work (a normal gated cwd write); it reloads next session. No new
   dependency. This *durable working memory* is distinct from the in-session
   `InMemorySaver` (which stays, solely to drive interrupt/resume within a
   session).
7. **Subagents (`task`) — full tools, gated, gateway-bound.** Pass one
   general-purpose subagent to `create_deep_agent(subagents=[…])` under
   `--files`. It **omits `model`** (inherits the gateway-bound model —
   `spec.get("model", model)` + `resolve_model` passes instances through, keeping
   the live agent AssemblyAI-only) and inherits the full toolset against the same
   sandboxed backend, with its own `interrupt_on` mirroring `_WRITE_TOOLS` so its
   `write_file`/`edit_file`/`execute` also prompt. **Verification-gated (see
   Architecture): whether a subagent's HITL interrupt surfaces through our
   approval loop is unverified; if implementation can't prove it, the subagent
   falls back to a read-only toolset — never an ungated mutating subagent.**
8. **Spoken approval — voice or keyboard, fail-safe to reject.** During an
   approval pause the agent accepts an **unambiguous spoken token** (an explicit
   phrase like "yes, run it" / "approve" — never a bare "yes", which STT
   mishears) **or** a keypress, whichever comes first. Anything ambiguous —
   silence, a timeout, a low-confidence or unrecognized utterance — **rejects**.
   For the highest-risk tier (commands `risk.py` flags as destructive), spoken
   approval is **not** accepted; those require the keyboard.

### Why these, over the alternatives (rejected)

- **Ephemeral scratch dir / fully isolated from cwd** — rejected: that's "run
  arbitrary code safely," not cowork. Confining writes to `/tmp` and deny-reading
  cwd means `execute` can't `pytest` the repo or build the files just edited.
- **`execute` unprompted (trust the sandbox alone)** — rejected: even confined
  to cwd, an approved-by-default agent could delete or rewrite project files; an
  approval keeps the human in the loop, with the sandbox limiting blast radius.
- **Keyboard-only approval** — rejected: a voice cowork agent whose safety gate
  requires the keyboard is a contradiction; spoken approval resolves it (with the
  destructive-tier keyboard fallback as the safety floor).
- **Enumerate-the-allowed-system-read-paths (deny reads by default)** —
  rejected: hand-maintaining the `/usr` / `/System` / `/Library` set a Python
  install needs is the most fragile part of a macOS sandbox. srt is
  default-allow-reads + deny-secrets for exactly this reason; we adopt that.
- **Persistent sqlite checkpointer for cross-session resume** — rejected: re-adds
  the deliberately-removed `langgraph-checkpoint-sqlite` dep, and the fresh-
  `thread_id`-per-turn design (avoids re-accumulating history) means thread-state
  resume doesn't map cleanly. `MemoryMiddleware` fits better.
- **`LocalShellBackend` unconfined** — rejected: deepagents itself warns it gives
  no isolation; the sandbox is the whole point.
- **Docker / container sandbox** — rejected: heavy daemon dependency, slow
  per-session cold start for a keyless CLI voice agent.
- **Depend on `@anthropic-ai/sandbox-runtime` directly** — rejected: Node-only,
  adds an `npx` runtime dependency. We borrow its posture, not its code.

## Scope

- **Live-only.** All new code lives in `aai_cli/agent_cascade/`; gated behind
  `--files`. Nothing else in the CLI changes.
- **No new dependency.** Pure subprocess over OS binaries; `MemoryMiddleware`,
  `InMemorySaver`, and the subagent middleware are already available.
- **Speakable contract preserved.** `execute` never raises into the graph; on
  any failure it returns a short string for the agent to speak.

### Out of scope (YAGNI)

- Windows sandboxing → `execute` stays inert there.
- Network access or package installation inside the sandbox.
- Docker / remote / cloud sandboxes.
- Full-transcript checkpointer resume; global/cross-project memory (memory is
  per-project, in cwd).
- A separate `--sandbox`/`--exec` flag or per-tool opt-outs.
- A richer command-risk *tiering* model beyond the two tiers we use (gated vs
  `risk.py`-flagged-destructive); the existing `risk.py` heuristic is the line.

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

- **The denylists (shared constants):** one read-deny tuple (credential stores +
  `.env*` + `.claude/`) and one within-cwd write-deny tuple (`.git/hooks/`, shell
  rc files), cribbed from srt's auto-protected set. Both renderers consume the
  same constants so the platforms stay in lockstep (a parity test asserts it).

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
    read-only = default-allow-reads), `--bind <cwd> <cwd>` (rw), `--bind <tmp>
    <tmp>`, `--chdir <cwd>`, then `--tmpfs`/`--ro-bind /dev/null` masks over each
    secret path and `--ro-bind` over `.git/hooks` to block writes; network
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
  `cwd`, `timeout`, minimal env via `core/env.child_env`) and the capability
  probe — both injectable so the suite asserts *what argv/profile we'd run* with
  no real sandbox (CI reliably has neither binary).

### Subagents (the `task` tool)

One general-purpose subagent, passed to `create_deep_agent(subagents=[spec])`
under `--files`. (deepagents exports a built-in `GENERAL_PURPOSE_SUBAGENT`, but
we define our own spec to set `interrupt_on` and omit `model`.) The spec (a
deepagents `SubAgent` dict):

- `name`: `"general-purpose"`; `description`: what `task()` is for (delegate a
  focused multi-step subtask — research, gather context, or implement a
  contained change — and get back a short summary).
- `system_prompt`: the cowork rules + "return a concise spoken-length summary."
- **`model`: omitted** — inherits the gateway-bound model. A test asserts the
  spec carries no `model` key so the AssemblyAI-only invariant can't silently
  regress to a `provider:model` string.
- **`tools`: omitted** in the full-tools path — inherits the main toolset
  (`read_file`/`write_file`/`edit_file`/`ls`/`glob`/`grep`/`execute`) bound to
  the same `SandboxedShellBackend`, so `execute` stays sandboxed in the subagent.
- **`interrupt_on`: `dict.fromkeys(_WRITE_TOOLS, True)`** — the subagent gets its
  own `HumanInTheLoopMiddleware` so its `write_file`/`edit_file`/`execute` also
  prompt (deepagents adds this when `interrupt_on` is set; it "Requires a
  checkpointer", which the `--files` graph already has).

**The verification gate (the one genuine unknown).** A subagent's HITL interrupt
is raised inside the subagent's sub-graph; our approval loop (`_stream_gated` →
`_pending_writes`) reads `graph.get_state(config).interrupts` at the *parent*
level. Whether a subagent interrupt surfaces there is **unverified**.
Implementation MUST prove it with a focused test/spike *before* shipping the
full-tools subagent. **If it does not surface, fall back to a read-only subagent
`tools` list** (`read_file`/`ls`/`glob`/`grep` + the keyless live tools, no
mutation/`execute`) — a researcher that can't bypass the gate. Shipping an
ungated mutating subagent is **not** acceptable; the read-only fallback is the
safety floor.

### Spoken approval (hands-free gating)

Today the `Approver` (`brain.Approver = Callable[[str, dict], bool]`) is answered
by `modals.ApprovalScreen`'s keypress. Spoken approval makes the *answer* source
multimodal without changing the gate's shape:

- During an `ApprovalPause(active=True)` (the reply deadline is already
  suspended for the human-think interval), the engine — whose STT stream is
  already live — races the **next final transcript** against a keypress and
  resolves the approver with whichever lands first.
- **Token grammar (fail-safe to reject):** approval requires an explicit
  affirmative phrase (e.g. "yes, run it" / "approve" / "go ahead and run it") —
  never a bare "yes" (STT confuses "no"/"go"/"yeah"). Negatives, low-confidence
  or unrecognized utterances, silence, and the pause timeout all resolve to
  **reject** (the existing `_DECLINED` path).
- **Destructive tier → keyboard only.** When `risk.risk_warning(name, args)`
  fires (the destructive-shell heuristic), the spoken affirmative is ignored and
  the prompt requires the keyboard, so an STT mishearing can never green-light an
  `rm -rf`/`sudo`/disk-write.
- **Boundaries touched:** this is the larger lift. It extends the approver
  protocol to a voice-aware variant the engine supplies (racing STT vs. keypress,
  with the risk-tier branch), and threads the "next spoken token" from
  `engine`'s STT leg into the approval window. The keyboard `ApprovalScreen`
  stays as the fallback and the headless `_deny_writes` auto-reject is unchanged.
  The STT/voice race is injected (a fake "spoken token" source) so it stays
  hermetic — no mic, no sockets.

### Edits to `brain.py` (the one shared file, minimal + additive)

- `_build_fs_backend()` returns `SandboxedShellBackend(root_dir=str(Path.cwd()),
  virtual_mode=True)` instead of `FilesystemBackend`. `--files`-off path
  unchanged.
- `_WRITE_TOOLS` becomes `("write_file", "edit_file", "execute")` so `execute`
  joins `interrupt_on` and the existing approval/resume loop
  (`_stream_gated`/`_decide`). The `InMemorySaver` checkpointer is unchanged.
- `_graph_kwargs` (when `config.files`) also sets `memory=["./.deepagents/
  AGENTS.md"]` and `subagents=[<the general-purpose spec>]` on `create_deep_agent`.
- `_TOOL_LABELS["execute"] = "Running code"` and
  `_TOOL_LABELS["task"] = "Working on a subtask"` — the live-UI affordances.
- The system-prompt capability phrasing advertises *"run code to solve problems
  and operate on this project"* when `execute` is bound, and *"delegate a bigger
  job to a helper"* when `task` is bound.

## Boundary / housekeeping

- `subprocess` is fenced by ruff `TID251`; `sandbox.py` gets a deliberate,
  reviewable per-module allowlist entry. The child env is built minimally via
  `core/env.child_env`.
- `risk.py`'s `execute` branch becomes **live** (shell-risk warning on the
  prompt; also the destructive-tier signal for the keyboard fallback) — no longer
  dormant, so its tests assert real behavior.
- Stale comments to fix: the "always-bound `execute` … inert" notes in
  `brain.py`; the `--files` paragraph in `aai_cli/CLAUDE.md` (now: sandboxed
  gated code execution + durable memory + delegation + voice approval); the
  `--files` help string (regenerate the affected `--help` snapshot; never
  hand-edit `.ambr`).
- Memory file lives at `./.deepagents/AGENTS.md` (deepagents convention). No new
  env var / command ⇒ docs-consistency gate stays green; update REFERENCE.md/
  README only if their `--files` description needs it.

## Error handling (cross-cutting)

`execute` is best-effort and never raises into the graph:

- capability `none` → *"I can't run code on this system."*
- sandbox launch failure (`Runner` raises) → a short apology string.
- timeout / non-zero exit → returned as combined output + `exit_code` for the
  model to read aloud (a failed run is information, not an error path).
- user declines (keypress, spoken negative, ambiguity, or timeout) → the standard
  `_DECLINED` message, same as a declined write today.

This mirrors the never-raise contract every live tool follows.

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks. Hermetic via the injected
`Runner`, capability, and spoken-token seams — no real sandbox, mic, or sockets.

- **Policy renderers:** `render_seatbelt_profile` asserts `(deny default)` +
  `(allow file-read*)`, each read-deny path emits a `file-read*` deny, **cwd is a
  `file-write*` subpath**, each write-deny path (incl. `.git/hooks`) emits a
  `file-write*` deny, and no network allow exists; `build_bwrap_argv` asserts
  `--unshare-all`, `--ro-bind / /`, the cwd rw bind, `--chdir <cwd>`, the secret
  masks, and the `.git/hooks` read-only bind. A **parity test** asserts both
  renderers cover the same denylist constants. Mutating any allow/deny token, or
  dropping a denylist entry, must fail a test.
- **`execute()` happy path:** a fake `Runner` asserts the command is wrapped in
  `sandbox-exec -p <profile>` / `bwrap …` with `cwd=<real cwd>`; timeout
  passthrough; output/exit shaping into `ExecuteResponse`.
- **Capability `none`:** asserts the refusal string **and that the `Runner` is
  never invoked** — kills the "fall back to host shell" mutant.
- **Failure modes:** `Runner` raising → apology; non-zero exit → output+exit.
- **brain wiring:** `_build_fs_backend()` returns a `SandboxBackendProtocol`
  backend; `execute` **is** in the `--files` `interrupt_on` map and a declined
  `execute` yields `_DECLINED`; `_tool_label("execute")` returns the new label;
  the capability phrase appears when `execute` is bound; `memory=` is passed with
  the per-project source when `--files` is on.
- **subagent wiring:** with `--files`, `create_deep_agent` gets a `subagents`
  list; the spec **carries no `model` key**; in the full-tools path its
  `interrupt_on` includes `execute`/`write_file`/`edit_file`; `_tool_label("task")`
  returns the new label; the `task` capability phrase appears when bound.
- **subagent HITL surfacing (the verification spike):** drive a subagent
  `write_file`/`execute` and assert it pauses through the parent approval loop
  (interrupt visible to `_pending_writes`). **Go/no-go for the full-tools
  subagent** — if it can't pass, the implementation switches to the read-only
  `tools` list and the test instead asserts the subagent has no mutating tools.
- **spoken approval:** an explicit affirmative phrase approves; a bare "yes",
  a negative, an unrecognized utterance, and a timeout each **reject**; a keypress
  still approves; and a `risk.py`-flagged destructive command **ignores** the
  spoken affirmative and requires the keypress. Drive via the injected spoken-token
  seam; assert the resolved decision, not mere execution.
- **`risk.py`:** the now-live branch asserts the warning fires for a destructive
  command and is `None` for a benign one (also exercised by the destructive-tier
  spoken-approval test).

## Milestones

Each is its own dependency-free PR; later milestones build on earlier ones.

- **M1 — Sandboxed `execute` + memory.** `sandbox.py`, the `brain.py` backend
  swap, `execute` in the gate (keyboard approval, the existing path), the
  `memory=` wiring, comment/help/doc updates, and their tests. The core value;
  shippable alone.
- **M2 — Subagents (`task`).** The general-purpose subagent + the HITL-surfacing
  spike that decides full-tools vs. read-only. Independent of M3.
- **M3 — Spoken approval.** The voice-aware approver, the engine STT-vs-keypress
  race, the token grammar, and the destructive-tier keyboard fallback — the
  largest lift, touching `engine`/`modals`. Makes M1/M2's gate hands-free.
