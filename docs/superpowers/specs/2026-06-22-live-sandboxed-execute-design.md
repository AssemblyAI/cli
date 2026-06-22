# Sandboxed `execute` for `assembly live`

**Date:** 2026-06-22
**Status:** Approved design — ready for implementation plan

## Goal

Let the `assembly live` voice agent (the `agent-cascade` command) **run code to
solve problems** — compute a number, parse some data, test an algorithm — by
lighting up deepagents' built-in `execute` tool. Today that tool is bound but
inert: `--files` uses a plain `FilesystemBackend`, which is not a
`SandboxBackendProtocol`, so `execute` only returns an error. We make `execute`
real, but confine it to an OS-kernel-isolated, throwaway workspace so a spoken
turn can run arbitrary shell **without** a confirmation prompt and without any
risk to the user's machine or files.

## Context

`assembly live` answers each spoken turn with a deepagents graph
(`aai_cli/agent_cascade/brain.py`). Tools are normally auto-approved — a
low-latency spoken turn can't pause for a keyboard confirmation. The `--files`
flag is the one exception: it swaps the in-memory backend for a real-cwd
`FilesystemBackend(virtual_mode=True)` and gates `write_file`/`edit_file` behind
a TUI `y/a/n` approval (`brain._stream_gated` + `agent_cascade.modals`). Reads
(incl. `grep`) stay ungated.

deepagents adds the `execute` tool automatically when the backend implements
`SandboxBackendProtocol`; for non-sandbox backends the tool returns an error
("inert"). The shipped options are `LocalShellBackend` (unrestricted host shell
— deepagents explicitly warns against untrusted/auto-approved use) or a
`BaseSandbox` subclass that implements `execute()` against real isolation. The
codebase already anticipates `execute`: `brain.py` comments call it
"always-bound … inert", and `risk.py` carries dormant shell-risk scoring for it.

There is **no first-class Python library** for macOS sandboxing. The idiomatic
mechanism is `sandbox-exec -p '<SBPL profile>' <command>` (Apple Seatbelt,
still shipping on current macOS, used by AI coding-agent sandboxes); on Linux
the equivalent is the `bwrap` (bubblewrap) binary. Both are pure-subprocess
patterns — no new dependency — which fits this repo (it already shells out to
controlled subprocesses; `S603/S607` are ignored project-wide for this).

## Decisions

1. **Isolation:** OS-level sandbox. `sandbox-exec -p '<SBPL>'` on macOS,
   `bwrap` on Linux. **Inert (safe refusal) on every other platform or when the
   sandbox binary is missing — never a fallback to unconfined execution.**
2. **Scope:** general shell — deepagents' native `execute(command)`.
3. **Activation:** folded into the existing `--files` flag (no new flag).
4. **Workspace:** file tools stay rooted at the **real cwd** (unchanged from
   today); `execute` runs in an **ephemeral `/tmp/aai-live-XXXX`**, time-bounded
   and deleted on session exit. **Read posture (cribbed from
   `@anthropic-ai/sandbox-runtime`): reads allowed by default so interpreters
   work, with cwd, `$HOME`, and a secrets denylist explicitly blocked**; writes
   permitted **only** under scratch; **no network**. So executed code can see
   system libraries and its own scratch, but never the user's project or
   credentials.
5. **Gating:** keep today's TUI approval for `write_file`/`edit_file` (they
   touch real files); `execute` runs **unprompted** — the sandbox is the
   boundary.

### Why these, over the alternatives (rejected)

- **`LocalShellBackend` unconfined + approve every `execute`** — rejected:
  approving shell commands by voice/TUI is clumsy, and deepagents itself warns
  the backend gives no isolation. The sandbox lets us drop the friction safely.
- **Docker / container sandbox** — rejected: a heavy daemon dependency and slow
  per-session cold start for a keyless CLI voice agent.
- **`execute` reads the real cwd (read-only)** — rejected: the executed code
  must not see the user's project or credentials. Note the read posture is
  still *default-allow* (so interpreters find their system libraries without us
  enumerating them), but cwd / `$HOME` / a secrets denylist are explicitly
  blocked. The model copies any needed data into the scratch workspace instead.
- **Enumerate-the-allowed-system-read-paths (deny reads by default)** —
  rejected after comparing to `@anthropic-ai/sandbox-runtime`: hand-maintaining
  the exact `/usr` / `/System` / `/Library` set a Python install needs is the
  most fragile part of a macOS sandbox and breaks across interpreters. srt
  (Anthropic's own Seatbelt/bwrap sandbox) is default-allow-reads +
  deny-the-sensitive-paths for exactly this reason; we adopt that posture.
- **Depend on `@anthropic-ai/sandbox-runtime` directly** — rejected: it is
  Node/TypeScript (CLI + JS library only, no Python binding), so using it adds
  a Node + `npx` runtime dependency for `execute`, cutting against the agent's
  keyless/no-setup ethos. We borrow its *posture* and profile lessons but keep
  a dependency-free pure-`subprocess` implementation over `sandbox-exec`/`bwrap`.

## Scope

- **Live-only.** All new code lives in `aai_cli/agent_cascade/`; the change is
  gated behind `--files`. Nothing else in the CLI changes.
- **No new dependency.** Pure subprocess over OS-provided binaries.
- **Speakable contract preserved.** `execute` never raises into the graph; on
  any failure it returns a short string for the agent to speak.

### Out of scope (YAGNI)

- Windows sandboxing → `execute` stays inert there.
- Docker / remote / cloud sandboxes.
- Network access or package installation inside the sandbox.
- Persisting the scratch workspace across sessions or turns.
- Per-tool opt-out flags; a separate `--sandbox`/`--exec` flag.

## Architecture

### New module: `aai_cli/agent_cascade/sandbox.py`

The entire sandbox concern in one focused, independently-testable module.

- **`class SandboxedShellBackend(LocalShellBackend)`** — inherits
  `FilesystemBackend` file operations rooted at cwd (so
  `read_file`/`write_file`/`edit_file`/`ls`/`glob`/`grep` behave exactly as
  `--files` does today) and **overrides `execute()`** so it never delegates to
  the inherited host-shell `execute`. Implementing `SandboxBackendProtocol` (via
  `LocalShellBackend`) is what makes deepagents auto-add the `execute` tool.
  - `execute(command, *, timeout=None) -> ExecuteResponse`: resolve capability →
    render the policy → run the wrapped command through the injected `Runner`
    with `cwd=<scratch>` → return combined stdout + exit code as
    `ExecuteResponse`. Bounded by `timeout` (default + a hard max).
  - **Invariant:** the override must never call `super().execute()` (the host
    shell). When capability is `none` it returns a refusal and does not run
    anything.

- **The secrets denylist (one shared constant):** the paths blocked from reads
  even under the default-allow posture, cribbed from srt's auto-protected set —
  the cwd, `$HOME` (broadly), `~/.ssh`, `~/.aws`, `~/.config`, `.env` files,
  `.git/`, `.claude/`, and shell rc files (`.bashrc`/`.zshrc`/`.profile`). One
  module-level tuple feeds both renderers so the two platforms stay in lockstep
  (a test asserts parity).

- **Policy rendering (pure functions — the security core):**
  - `render_seatbelt_profile(scratch: str, *, deny_read: Sequence[str]) -> str`
    — SBPL string with a **default-allow-reads** posture: `(version 1)`,
    `(deny default)`, `(allow process-exec*)`, `(allow file-read*)` then
    `(deny file-read* (subpath …))` for each denylist entry (last-match-wins, so
    the denies override the blanket allow), `(allow file-write* (subpath
    "<scratch>"))`, and network left denied by `(deny default)`. cwd / `$HOME`
    / secrets appear **only** in the deny rules.
  - `build_bwrap_argv(scratch, command, *, deny_read) -> list[str]` —
    `bwrap --unshare-all --die-with-parent`, `--ro-bind / /` (the whole FS
    read-only, the Linux equivalent of default-allow-reads), then a `--tmpfs`
    mask over each denylist path (cwd, `$HOME`, …) so they read as empty,
    `--bind <scratch> <scratch>` read-write as the working dir, network
    unshared. The tmpfs-masking is how "read everything except these" is
    expressed in bubblewrap's bind-mount model.
  - **Optional hardening:** wrap the inner command with `ulimit -t` (CPU
    seconds) and `ulimit -v` (address space) so a runaway computation can't peg
    the machine even inside the wall-clock timeout. Mark the literal caps
    `# pragma: no mutate` (tuning knobs).

- **Capability detection (injectable):** resolve `"seatbelt" | "bwrap" |
  "none"` from the platform plus a `which`-style probe for the binary. `"none"`
  → `execute` returns *"I can't run code on this system."* and **never** shells
  out. This refuse-don't-fall-back branch is the single most safety-critical
  line in the feature.

- **Seams for hermetic tests:**
  - `Runner = Callable[[list[str], str, int], CompletedProcessLike]` — default
    wraps `subprocess.run` (combined output, `cwd`, `timeout`, minimal env).
  - the capability probe — injectable so a test can force seatbelt/bwrap/none
    regardless of the host. CI reliably has neither binary, so the suite asserts
    *what argv/profile we would run*, never a real sandbox.

- **Scratch lifecycle:** `tempfile.mkdtemp(prefix="aai-live-")` once per backend
  instance; removed when the session ends.

### Edits to `brain.py` (the one shared file, minimal + additive)

- `_build_fs_backend()` returns `SandboxedShellBackend(root_dir=str(Path.cwd()),
  virtual_mode=True)` instead of `FilesystemBackend`. The `--files`-off path is
  unchanged. `_WRITE_TOOLS` stays `("write_file", "edit_file")` — `execute` is
  deliberately **not** added to `interrupt_on`, so it is auto-approved.
- `_TOOL_LABELS["execute"] = "Running code"` — the live-UI affordance shown
  while a code run is in flight.
- The system-prompt capability phrasing advertises *"run code to solve
  problems"* only when `execute` is in the bound toolset.

## Boundary / housekeeping

- `subprocess` is fenced by ruff `TID251`; `sandbox.py` gets a deliberate,
  reviewable per-module allowlist entry (the established pattern). The child env
  is built minimally via `core/env.child_env`.
- Stale comments to fix: the "always-bound `execute` … inert" notes in
  `brain.py` (`_WRITE_TOOLS` block and `_build_fs_backend`), the `--files`
  paragraph in `aai_cli/CLAUDE.md`, and the `--files` help string (regenerate
  the affected `--help` snapshot; never hand-edit `.ambr`).
- No new env var or command ⇒ the docs-consistency gate stays green (verify
  during implementation; update REFERENCE.md/README only if the `--files`
  description there mentions code execution).
- `risk.py` already scores `execute`; since `execute` is ungated its warning is
  dormant — left as-is, not removed.

## Error handling (cross-cutting)

`execute` is best-effort and never raises into the graph:

- capability `none` → *"I can't run code on this system."*
- sandbox launch failure (`Runner` raises) → a short apology string.
- timeout / non-zero exit → returned as combined output + `exit_code` for the
  model to read aloud (a failed run is information, not an error path).

This mirrors the never-raise contract every live tool follows, so a sandbox
problem can't trip `brain`'s "couldn't complete the turn" path.

## Testing

Targets the gate's 100% patch-coverage + diff-scoped mutation requirements:
assertions must *fail* if a changed line breaks, not merely execute it. One
`tests/test_agent_cascade_sandbox.py`, fully hermetic via the injected `Runner`
and capability seams — no real sandbox, no sockets.

- **Policy renderers:** `render_seatbelt_profile` asserts `(deny default)` +
  `(allow file-read*)` present (default-allow reads), every denylist path emits
  a `(deny file-read* (subpath …))`, `scratch` is the **only** `file-write*`
  subpath, and no network allow rule exists; `build_bwrap_argv` asserts
  `--unshare-all`, `--ro-bind / /`, a `--tmpfs` mask for each denylist path, and
  the scratch rw bind as workdir. A **parity test** asserts both renderers cover
  the same denylist constant. Mutating any allow/deny token, or dropping a
  denylist entry, must fail a test.
- **`execute()` happy path:** a fake `Runner` asserts the command is wrapped in
  `sandbox-exec -p <profile>` (seatbelt) / `bwrap …` (bwrap) with `cwd=scratch`;
  timeout passthrough; combined output + exit-code shaping into
  `ExecuteResponse`.
- **Capability `none`:** asserts the refusal string **and that the `Runner` is
  never invoked** — kills the "fall back to host shell" mutant.
- **Failure modes:** `Runner` raising → apology; non-zero exit → output+exit
  surfaced.
- **brain wiring:** `_build_fs_backend()` returns a backend that satisfies
  `SandboxBackendProtocol` (so deepagents binds `execute`); `execute` is absent
  from the `--files` `interrupt_on` map; `_tool_label("execute")` returns the
  new label; the capability phrase appears when `execute` is bound. These assert
  the exact behavior/string, not mere execution.

## PR sequence

**Single feature PR.** No new dependency, so no separate `uv.lock` PR is needed.
The change is `sandbox.py` + the `brain.py` wiring + comment/help/doc updates +
the tests.
