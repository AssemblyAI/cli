# Sandboxed `execute` + durable memory for `assembly live` (M1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `assembly live --files` able to run shell commands in the real cwd, OS-sandbox-confined (can't reach the network or escape cwd) and human-approved per run, and remember its work across sessions via a per-project memory file.

**Architecture:** One new focused module `aai_cli/agent_cascade/sandbox.py` holds the entire sandbox concern: a `SandboxedShellBackend(LocalShellBackend)` whose `execute()` never delegates to the unconfined host shell but instead wraps the command in an OS sandbox (`sandbox-exec` SBPL on macOS, `bwrap` on Linux, refuse on every other platform / missing binary), driven by pure policy renderers and injectable `Runner`/capability seams so the suite is hermetic. `brain.py` swaps its filesystem backend for this one (which makes deepagents auto-add a *functional* `execute` tool), adds `execute` to the existing approval gate, and turns on `MemoryMiddleware` via `memory=["./.deepagents/AGENTS.md"]`.

**Tech Stack:** Python 3.12+, deepagents + langgraph, Typer/Textual, pytest + syrupy, `uv`. Pure-subprocess OS sandbox — **no new dependency**.

**Spec:** `docs/superpowers/specs/2026-06-22-live-sandboxed-execute-design.md` (this plan implements **Milestone M1** only — "Sandboxed `execute` + memory". M2 subagents and M3 spoken approval are separate plans/PRs that build on this one.)

## Global Constraints

- `from __future__ import annotations` at the top of every module (verbatim from `CLAUDE.md`).
- **No new dependency.** Pure subprocess over OS binaries (`sandbox-exec`, `bwrap`); `MemoryMiddleware` and `InMemorySaver` are already available. Do **not** touch `uv.lock`.
- **Live-only.** All new code lives in `aai_cli/agent_cascade/`; everything is gated behind `--files`. The `--files`-off path must be byte-for-byte unchanged.
- **Inert (safe refusal), never a fallback to unconfined execution.** On any platform other than macOS/Linux, or when the sandbox binary is missing, capability is `"none"` and `execute` returns a refusal string and **runs nothing**. The override must **never** call `super().execute()` (the unconfined host shell). This refuse-don't-fall-back branch is the single most safety-critical line.
- **`execute` never raises into the graph** — on any failure it returns a short string for the model to speak (the never-raise contract every live tool follows).
- **Errors → stderr, data → stdout** (repo invariant; not directly relevant here but preserve it).
- **Help copy is terse, imperative, sentence-case, no trailing period** (Codex-CLI style). Help strings are pinned by syrupy `--help` goldens — regenerate with `--snapshot-update`, never hand-edit `.ambr`.
- **`subprocess` is fenced by ruff `TID251`.** `sandbox.py` needs a deliberate per-file allowlist entry in `pyproject.toml`. Build the child env via `aai_cli/core/env.child_env`.
- **Max file length is 500 lines** (`scripts/max_file_length.py`) — keep `sandbox.py` under it.
- **Gate reality (from memory + `CLAUDE.md`):** `./scripts/check.sh` enforces **100% patch coverage vs `origin/main`** *and* a **diff-scoped mutation gate** — a changed boolean/string/branch survives unless a test asserts the behavioral *difference* between its two values, not merely that the line ran. There is also a **no-new-escape-hatches** gate (no net-new `# type: ignore`/`# noqa`/`pragma: no cover`/`Any`/`cast(`/test skip/xfail/sleep vs merge-base). `# pragma: no mutate` is the sanctioned way to exempt a genuinely unassertable tuning literal (use it for the `ulimit` caps only).
- **Commit hook:** a PreToolUse hook blocks `git commit` unless `./scripts/check.sh` last passed for the current working-tree signature. Use `AAI_ALLOW_COMMIT=1 git commit …` for the per-task WIP commits below, then run the **full** `./scripts/check.sh` once at the end (Task 7) and let that gate the final state.
- **Workspace:** execute on the current `live-tool-call-ux` branch (it already carries this feature's design docs). Commit ONLY this feature's files; never `git add -A`.

---

### Task 1: Denylist constants + Seatbelt profile renderer (`sandbox.py`)

The security core, part 1. Pure function, no I/O — fully unit-testable.

**Files:**
- Create: `aai_cli/agent_cascade/sandbox.py`
- Test: `tests/test_agent_cascade_sandbox.py`

**Interfaces:**
- Produces:
  - `HOME_SECRETS: tuple[str, ...]` — credential dirs/files relative to `$HOME` (`.ssh`, `.aws`, `.gnupg`, `.netrc`, `.npmrc`).
  - `CWD_READ_DENY: tuple[str, ...]` — project-local secrets denied for reads even though cwd is otherwise readable (`.env`, `.claude`). `.env` also covers `.env.*`.
  - `CWD_WRITE_DENY: tuple[str, ...]` — persistence paths denied for writes even inside cwd (`.git/hooks`).
  - `SHELL_RC: tuple[str, ...]` — shell rc files denied for writes (matters only when `cwd == $HOME`): `.bashrc`, `.zshrc`, `.profile`, `.bash_profile`.
  - `render_seatbelt_profile(cwd: str, tmp: str, home: str, *, home_secrets: Sequence[str] = HOME_SECRETS, cwd_read_deny: Sequence[str] = CWD_READ_DENY, cwd_write_deny: Sequence[str] = CWD_WRITE_DENY, shell_rc: Sequence[str] = SHELL_RC) -> str` — an SBPL profile string.

- [ ] **Step 1: Write the failing test**

Add to a new `tests/test_agent_cascade_sandbox.py`:

```python
from __future__ import annotations

from aai_cli.agent_cascade import sandbox


def test_seatbelt_profile_is_default_allow_reads_deny_default():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process-exec*)" in profile
    assert "(allow file-read*)" in profile  # default-allow reads
    # No network allow anywhere — network stays denied by (deny default).
    assert "network" not in profile


def test_seatbelt_profile_denies_each_home_secret_for_reads():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    for name in sandbox.HOME_SECRETS:
        assert f'(deny file-read* (subpath "/home/u/{name}"))' in profile


def test_seatbelt_profile_denies_project_secrets_for_reads():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    # .env (and .env.*) under cwd are read-denied via a regex; .claude/ via subpath.
    assert "file-read*" in profile and "/work/proj" in profile
    assert any(".env" in line and "deny file-read*" in line for line in profile.splitlines())
    assert '(deny file-read* (subpath "/work/proj/.claude"))' in profile


def test_seatbelt_profile_writes_confined_to_cwd_and_tmp():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert '(allow file-write* (subpath "/work/proj") (subpath "/tmp"))' in profile


def test_seatbelt_profile_denies_persistence_writes_inside_cwd():
    profile = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    assert '(deny file-write* (subpath "/work/proj/.git/hooks"))' in profile
    # Shell rc files denied for writes (covers the cwd == $HOME case).
    for name in sandbox.SHELL_RC:
        assert f'(deny file-write* (subpath "/home/u/{name}"))' in profile
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: FAIL — `ModuleNotFoundError: aai_cli.agent_cascade.sandbox` (or `AttributeError`).

- [ ] **Step 3: Write the minimal implementation**

Create `aai_cli/agent_cascade/sandbox.py`:

```python
"""OS-sandboxed shell execution for ``assembly live --files``.

deepagents binds a functional ``execute`` tool only when the backend implements
``SandboxBackendProtocol``. :class:`SandboxedShellBackend` does — but its ``execute`` never
runs an unconfined host shell: it wraps the command in an OS sandbox (``sandbox-exec`` SBPL on
macOS, ``bwrap`` on Linux) that confines writes to cwd, denies the network, and read-denies
credential stores. On any other platform (or with the sandbox binary missing) it refuses and
runs nothing — never a fallback to unconfined execution. The policy renderers are pure and the
subprocess/capability boundaries are injected, so the suite asserts *what we would run* with no
real sandbox.
"""

from __future__ import annotations

from collections.abc import Sequence

# Credential dirs/files under $HOME, read-denied precisely on both platforms.
HOME_SECRETS: tuple[str, ...] = (".ssh", ".aws", ".gnupg", ".netrc", ".npmrc")
# Project-local secrets denied for reads even though cwd is otherwise readable.
CWD_READ_DENY: tuple[str, ...] = (".env", ".claude")
# Persistence paths denied for writes even inside cwd.
CWD_WRITE_DENY: tuple[str, ...] = (".git/hooks",)
# Shell rc files denied for writes (only inside the write region when cwd == $HOME).
SHELL_RC: tuple[str, ...] = (".bashrc", ".zshrc", ".profile", ".bash_profile")


def render_seatbelt_profile(
    cwd: str,
    tmp: str,
    home: str,
    *,
    home_secrets: Sequence[str] = HOME_SECRETS,
    cwd_read_deny: Sequence[str] = CWD_READ_DENY,
    cwd_write_deny: Sequence[str] = CWD_WRITE_DENY,
    shell_rc: Sequence[str] = SHELL_RC,
) -> str:
    """Render an Apple Seatbelt (SBPL) profile: default-allow reads, deny secrets, writes only
    in cwd + tmp, no network. Last-match-wins, so the denies override the broad allows."""
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec*)",
        "(allow process-fork)",
        "(allow file-read*)",
    ]
    for name in home_secrets:
        lines.append(f'(deny file-read* (subpath "{home}/{name}"))')
    # .env and .env.* under cwd, denied via regex; .claude/ via subpath.
    lines.append(f'(deny file-read* (regex #"^{cwd}/\\.env($|\\.)"))')
    for name in cwd_read_deny:
        if name == ".env":
            continue
        lines.append(f'(deny file-read* (subpath "{cwd}/{name}"))')
    lines.append(f'(allow file-write* (subpath "{cwd}") (subpath "{tmp}"))')
    for name in cwd_write_deny:
        lines.append(f'(deny file-write* (subpath "{cwd}/{name}"))')
    for name in shell_rc:
        lines.append(f'(deny file-write* (subpath "{home}/{name}"))')
    return "\n".join(lines) + "\n"
```

> Note: `CWD_READ_DENY` carries `.env` (rendered as the regex line) and `.claude` (rendered as a subpath). The test `test_seatbelt_profile_denies_project_secrets_for_reads` pins both; keep them in the constant so the parity test in Task 2 can assert both renderers cover the same set.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/sandbox.py tests/test_agent_cascade_sandbox.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): seatbelt sandbox profile renderer + denylist constants"
```

---

### Task 2: bwrap argv builder + parity test (`sandbox.py`)

The security core, part 2 (Linux), plus the parity test that keeps the two platforms in lockstep.

**Files:**
- Modify: `aai_cli/agent_cascade/sandbox.py`
- Test: `tests/test_agent_cascade_sandbox.py`

**Interfaces:**
- Produces: `build_bwrap_argv(cwd: str, tmp: str, command: str, home: str, *, home_secrets: Sequence[str] = HOME_SECRETS, cwd_read_deny: Sequence[str] = CWD_READ_DENY, cwd_write_deny: Sequence[str] = CWD_WRITE_DENY) -> list[str]` — the full `bwrap` argv ending in the shell invocation of `command`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_cascade_sandbox.py`:

```python
def test_bwrap_argv_confines_to_cwd_with_network_unshared():
    argv = sandbox.build_bwrap_argv("/work/proj", "/tmp", "echo hi", "/home/u")
    assert argv[0] == "bwrap"
    assert "--unshare-all" in argv  # includes network namespace
    assert "--die-with-parent" in argv
    # Whole FS read-only = default-allow reads.
    assert _has_pair(argv, "--ro-bind", "/", "/")
    # cwd + tmp are read-write bound; chdir into cwd.
    assert _has_pair(argv, "--bind", "/work/proj", "/work/proj")
    assert _has_pair(argv, "--bind", "/tmp", "/tmp")
    assert _adjacent(argv, "--chdir", "/work/proj")
    # The command lands at the tail via a shell.
    assert argv[-1] == "echo hi" or "echo hi" in argv[-1]


def test_bwrap_argv_masks_home_secrets_and_git_hooks():
    argv = sandbox.build_bwrap_argv("/work/proj", "/tmp", "echo hi", "/home/u")
    joined = " ".join(argv)
    for name in sandbox.HOME_SECRETS:
        assert f"/home/u/{name}" in joined  # masked (tmpfs / ro-bind /dev/null)
    assert "/work/proj/.git/hooks" in joined  # write blocked via ro-bind


def _has_pair(argv, flag, a, b):
    for i in range(len(argv) - 2):
        if argv[i] == flag and argv[i + 1] == a and argv[i + 2] == b:
            return True
    return False


def _adjacent(argv, flag, value):
    for i in range(len(argv) - 1):
        if argv[i] == flag and argv[i + 1] == value:
            return True
    return False


def test_renderers_cover_the_same_denylists():
    # Parity: both platform renderers must reference every denylist constant, so a path added
    # to one platform can't silently be left unprotected on the other.
    seatbelt = sandbox.render_seatbelt_profile("/work/proj", "/tmp", "/home/u")
    bwrap = " ".join(sandbox.build_bwrap_argv("/work/proj", "/tmp", "x", "/home/u"))
    for name in sandbox.HOME_SECRETS:
        assert f"/home/u/{name}" in seatbelt
        assert f"/home/u/{name}" in bwrap
    assert "/work/proj/.git/hooks" in seatbelt
    assert "/work/proj/.git/hooks" in bwrap
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_bwrap_argv'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `aai_cli/agent_cascade/sandbox.py`:

```python
def build_bwrap_argv(
    cwd: str,
    tmp: str,
    command: str,
    home: str,
    *,
    home_secrets: Sequence[str] = HOME_SECRETS,
    cwd_read_deny: Sequence[str] = CWD_READ_DENY,
    cwd_write_deny: Sequence[str] = CWD_WRITE_DENY,
) -> list[str]:
    """Build a bubblewrap argv: whole FS read-only (default-allow reads), cwd + tmp read-write,
    secret stores masked, ``.git/hooks`` read-only, network unshared. Path-based, so in-cwd
    secret-file protection is coarser than Seatbelt's globbing (a documented asymmetry); the
    directory-level credential stores are masked precisely on both."""
    argv = [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--ro-bind",
        "/",
        "/",
        "--bind",
        cwd,
        cwd,
        "--bind",
        tmp,
        tmp,
    ]
    # Mask credential stores under $HOME (tmpfs hides their contents).
    for name in home_secrets:
        argv += ["--tmpfs", f"{home}/{name}"]
    # Project-local secrets: mask each path (best-effort; coarser than Seatbelt).
    for name in cwd_read_deny:
        argv += ["--ro-bind", "/dev/null", f"{cwd}/{name}"]
    # Block writes to persistence paths inside cwd by re-binding them read-only.
    for name in cwd_write_deny:
        argv += ["--ro-bind", f"{cwd}/{name}", f"{cwd}/{name}"]
    argv += ["--chdir", cwd, "/bin/sh", "-c", command]
    return argv
```

> If a `--ro-bind /dev/null <path>` for a non-existent project secret makes `bwrap` error at launch, that surfaces as a `Runner` failure → apology string (Task 4), never a crash. The coarser-protection asymmetry is acknowledged in the spec.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/sandbox.py tests/test_agent_cascade_sandbox.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): bwrap argv builder + renderer parity test"
```

---

### Task 3: Capability detection + default Runner (`sandbox.py`)

The platform probe (`seatbelt | bwrap | none`) and the `subprocess` boundary, both injectable.

**Files:**
- Modify: `aai_cli/agent_cascade/sandbox.py` (+ `pyproject.toml` for the `TID251` allowlist)
- Test: `tests/test_agent_cascade_sandbox.py`

**Interfaces:**
- Produces:
  - `Capability = Literal["seatbelt", "bwrap", "none"]`
  - `detect_capability(*, system: Callable[[], str] = platform.system, which: Callable[[str], str | None] = shutil.which) -> Capability`
  - `class CompletedProcessLike(Protocol)` with `output: str` and `returncode: int | None`
  - `Runner = Callable[[list[str], str, int], CompletedProcessLike]`
  - `default_runner(argv: list[str], cwd: str, timeout: int) -> CompletedProcessLike` — wraps `subprocess.run` (combined stdout+stderr, `cwd`, `timeout`, env via `child_env`), returning partial output + a sentinel `returncode` on timeout instead of raising.
  - `DEFAULT_TIMEOUT_SECONDS: int`, `MAX_TIMEOUT_SECONDS: int`, `CPU_LIMIT_SECONDS: int`, `ADDRESS_LIMIT_KB: int`.

- [ ] **Step 1: Add the `TID251` allowlist entry to `pyproject.toml`**

In `[tool.ruff.lint.per-file-ignores]` (next to the existing `procs.py`/`coding_agent.py` entries) add:

```toml
# Sandbox shell-out: launches the OS sandbox binary (sandbox-exec / bwrap) with controlled
# argv; the whole module exists to confine that one subprocess call.
"aai_cli/agent_cascade/sandbox.py" = ["TID251"]
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_agent_cascade_sandbox.py`:

```python
def test_detect_capability_seatbelt_on_macos_with_binary():
    cap = sandbox.detect_capability(system=lambda: "Darwin", which=lambda _n: "/usr/bin/sandbox-exec")
    assert cap == "seatbelt"


def test_detect_capability_bwrap_on_linux_with_binary():
    cap = sandbox.detect_capability(system=lambda: "Linux", which=lambda _n: "/usr/bin/bwrap")
    assert cap == "bwrap"


def test_detect_capability_none_when_binary_missing():
    cap = sandbox.detect_capability(system=lambda: "Darwin", which=lambda _n: None)
    assert cap == "none"


def test_detect_capability_none_on_unsupported_platform():
    cap = sandbox.detect_capability(system=lambda: "Windows", which=lambda _n: "anything")
    assert cap == "none"
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q -k capability`
Expected: FAIL.

- [ ] **Step 4: Write the minimal implementation**

Append imports at the top of `sandbox.py` (keep `from __future__ import annotations` first):

```python
import platform
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from aai_cli.core.env import child_env
```

(Merge the `collections.abc` import with the existing `Sequence` one.) Then append:

```python
Capability = Literal["seatbelt", "bwrap", "none"]

DEFAULT_TIMEOUT_SECONDS = 120  # pragma: no mutate
MAX_TIMEOUT_SECONDS = 600  # pragma: no mutate
CPU_LIMIT_SECONDS = 60  # pragma: no mutate
ADDRESS_LIMIT_KB = 4_000_000  # pragma: no mutate
_TIMEOUT_EXIT = 124  # conventional timeout exit code


def detect_capability(
    *,
    system: Callable[[], str] = platform.system,
    which: Callable[[str], str | None] = shutil.which,
) -> Capability:
    """Resolve the sandbox mechanism for this host: ``seatbelt`` (macOS + ``sandbox-exec``),
    ``bwrap`` (Linux + ``bwrap``), else ``none`` — the refuse-don't-fall-back signal."""
    name = system()
    if name == "Darwin" and which("sandbox-exec"):
        return "seatbelt"
    if name == "Linux" and which("bwrap"):
        return "bwrap"
    return "none"


class CompletedProcessLike(Protocol):
    """The slice of a finished process the backend reads: combined output + exit code."""

    output: str
    returncode: int | None


class _Result:
    """Concrete :class:`CompletedProcessLike` the default runner returns."""

    def __init__(self, output: str, returncode: int | None) -> None:
        self.output = output
        self.returncode = returncode


Runner = Callable[[list[str], str, int], CompletedProcessLike]


def default_runner(argv: list[str], cwd: str, timeout: int) -> CompletedProcessLike:
    """Run ``argv`` with combined output, in ``cwd``, time-bounded, with a minimal child env.

    A timeout returns the partial output + a sentinel exit code (information, not a crash); a
    launch failure is left to raise so the caller turns it into an apology string."""
    try:
        proc = subprocess.run(  # noqa: S603 — argv is the controlled sandbox invocation
            argv,
            cwd=cwd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.output or ""
        text = out.decode() if isinstance(out, bytes) else out
        return _Result(text + f"\n[timed out after {timeout}s]", _TIMEOUT_EXIT)
    return _Result(proc.stdout or "", proc.returncode)
```

> The `S603` inline `# noqa` is pre-existing project policy (the repo ignores `S603/S607` project-wide for controlled shell-outs, per `CLAUDE.md`). If `ruff` reports it as unused because the rule is already globally ignored, drop the `# noqa` — do not add a net-new escape hatch (the no-escape-hatches gate counts these). Verify with `uv run ruff check aai_cli/agent_cascade/sandbox.py`.

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aai_cli/agent_cascade/sandbox.py tests/test_agent_cascade_sandbox.py pyproject.toml
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): sandbox capability probe + default subprocess runner"
```

---

### Task 4: `SandboxedShellBackend.execute()` (`sandbox.py`)

Wire the renderers + capability + runner into the backend override. This is where the never-call-`super().execute()` invariant lives.

**Files:**
- Modify: `aai_cli/agent_cascade/sandbox.py`
- Test: `tests/test_agent_cascade_sandbox.py`

**Interfaces:**
- Consumes: `render_seatbelt_profile`, `build_bwrap_argv`, `detect_capability`, `default_runner`, `Runner`, `Capability`, the timeout/limit constants (all from Task 1–3); `ExecuteResponse` from `deepagents.backends.protocol`; `LocalShellBackend` from `deepagents.backends.local_shell`.
- Produces: `class SandboxedShellBackend(LocalShellBackend)` with `__init__(self, *, root_dir: str, virtual_mode: bool = True, runner: Runner | None = None, capability: Capability | None = None, tmp: str | None = None, home: str | None = None)` and `execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse`.
- The refusal string constant `NO_SANDBOX_MESSAGE = "I can't run code on this system."` and `LAUNCH_FAILURE_MESSAGE = "I couldn't start a sandbox to run that."`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent_cascade_sandbox.py`:

```python
from deepagents.backends.protocol import ExecuteResponse


def _backend(tmp_path, cap, runner):
    return sandbox.SandboxedShellBackend(
        root_dir=str(tmp_path),
        capability=cap,
        runner=runner,
        tmp="/tmp",
        home="/home/u",
    )


def test_execute_seatbelt_wraps_command_in_sandbox_exec(tmp_path):
    calls = []

    def runner(argv, cwd, timeout):
        calls.append((argv, cwd, timeout))
        return sandbox._Result("done", 0)

    backend = _backend(tmp_path, "seatbelt", runner)
    resp = backend.execute("pytest -q", timeout=30)

    argv, cwd, timeout = calls[0]
    assert argv[0] == "sandbox-exec" and argv[1] == "-p"
    assert "(deny default)" in argv[2]  # the rendered profile
    assert "pytest -q" in argv[-1]  # command at the tail (ulimit-wrapped)
    assert cwd == str(tmp_path.resolve())
    assert timeout == 30
    assert isinstance(resp, ExecuteResponse)
    assert resp.output == "done" and resp.exit_code == 0


def test_execute_bwrap_uses_bwrap_argv(tmp_path):
    seen = {}

    def runner(argv, cwd, timeout):
        seen["argv"] = argv
        return sandbox._Result("ok", 0)

    _backend(tmp_path, "bwrap", runner).execute("ls")
    assert seen["argv"][0] == "bwrap"


def test_execute_capability_none_refuses_and_never_runs(tmp_path):
    # Record-and-assert-not-called (no `# pragma: no cover` — that's a gated escape hatch).
    calls = []

    def runner(argv, cwd, timeout):
        calls.append(argv)
        return sandbox._Result("", 0)

    resp = _backend(tmp_path, "none", runner).execute("rm -rf /")
    assert resp.output == sandbox.NO_SANDBOX_MESSAGE
    assert resp.exit_code is None
    assert calls == []  # the killer assertion: refusal must run nothing


def test_execute_never_calls_super_execute(tmp_path, monkeypatch):
    # The unconfined host shell must never run, even on the happy path. A one-line lambda
    # records the call so there's no never-executed function body to leave uncovered.
    from deepagents.backends.local_shell import LocalShellBackend

    super_calls = []
    monkeypatch.setattr(
        LocalShellBackend,
        "execute",
        lambda self, command, *, timeout=None: super_calls.append(command),
    )
    backend = _backend(tmp_path, "seatbelt", lambda a, c, t: sandbox._Result("x", 0))
    assert backend.execute("echo hi").output == "x"
    assert super_calls == []  # host shell never invoked


def test_execute_runner_failure_returns_apology(tmp_path):
    def runner(argv, cwd, timeout):
        raise OSError("sandbox-exec missing")

    resp = _backend(tmp_path, "seatbelt", runner).execute("echo hi")
    assert resp.output == sandbox.LAUNCH_FAILURE_MESSAGE
    assert resp.exit_code is None


def test_execute_nonzero_exit_passes_output_and_code_through(tmp_path):
    runner = lambda a, c, t: sandbox._Result("boom\n", 1)
    resp = _backend(tmp_path, "seatbelt", runner).execute("false")
    assert resp.output == "boom\n" and resp.exit_code == 1


def test_execute_clamps_timeout_to_max(tmp_path):
    seen = {}

    def runner(argv, cwd, timeout):
        seen["timeout"] = timeout
        return sandbox._Result("", 0)

    _backend(tmp_path, "seatbelt", runner).execute("x", timeout=10_000)
    assert seen["timeout"] == sandbox.MAX_TIMEOUT_SECONDS


def test_execute_defaults_timeout_when_unset(tmp_path):
    seen = {}
    _backend(tmp_path, "seatbelt", lambda a, c, t: (seen.update(t=t) or sandbox._Result("", 0))).execute("x")
    assert seen["t"] == sandbox.DEFAULT_TIMEOUT_SECONDS
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q -k execute`
Expected: FAIL — `SandboxedShellBackend` undefined.

- [ ] **Step 3: Write the minimal implementation**

Append to `aai_cli/agent_cascade/sandbox.py`:

```python
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

NO_SANDBOX_MESSAGE = "I can't run code on this system."
LAUNCH_FAILURE_MESSAGE = "I couldn't start a sandbox to run that."


def _ulimit_wrap(command: str) -> str:
    """Cap CPU + address space so a runaway can't peg the box inside the timeout."""
    return f"ulimit -t {CPU_LIMIT_SECONDS}; ulimit -v {ADDRESS_LIMIT_KB}; {command}"  # pragma: no mutate


class SandboxedShellBackend(LocalShellBackend):
    """A ``LocalShellBackend`` whose ``execute`` runs through an OS sandbox, never the host shell.

    Inherits the cwd-rooted file operations (``read_file``/``write_file``/``edit_file``/``ls``/
    ``glob``/``grep``) unchanged; implementing ``SandboxBackendProtocol`` (via the base) is what
    makes deepagents auto-add the ``execute`` tool. The override confines every run to cwd, denies
    the network, and refuses outright when no sandbox is available."""

    def __init__(
        self,
        *,
        root_dir: str,
        virtual_mode: bool = True,
        runner: Runner | None = None,
        capability: Capability | None = None,
        tmp: str | None = None,
        home: str | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, virtual_mode=virtual_mode)
        self._runner: Runner = runner or default_runner
        self._capability: Capability = capability if capability is not None else detect_capability()
        import os
        import tempfile

        self._tmp = tmp if tmp is not None else tempfile.gettempdir()
        self._home = home if home is not None else os.path.expanduser("~")

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` confined to cwd via the OS sandbox; refuse when none is available."""
        if self._capability == "none":
            return ExecuteResponse(output=NO_SANDBOX_MESSAGE, exit_code=None)
        cwd = str(self.cwd)
        wrapped = _ulimit_wrap(command)
        if self._capability == "seatbelt":
            profile = render_seatbelt_profile(cwd, self._tmp, self._home)
            argv = ["sandbox-exec", "-p", profile, "/bin/sh", "-c", wrapped]
        else:
            argv = build_bwrap_argv(cwd, self._tmp, wrapped, self._home)
        bounded = min(timeout or DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS)
        try:
            result = self._runner(argv, cwd, bounded)
        except Exception:  # noqa: BLE001 — any launch failure becomes a speakable apology
            return ExecuteResponse(output=LAUNCH_FAILURE_MESSAGE, exit_code=None)
        return ExecuteResponse(output=result.output, exit_code=result.returncode)
```

> Move the `import os` / `import tempfile` to module top (the post-edit ruff hook will not, since they're used immediately; cleaner to hoist them). The `# noqa: BLE001` is a net-new escape hatch — prefer catching `(OSError, ValueError, subprocess.SubprocessError)` instead of bare `Exception` so no `noqa` is needed and the no-escape-hatches gate stays green. Adjust the `test_execute_runner_failure_returns_apology` runner to raise `OSError` (already does).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_agent_cascade_sandbox.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Sanity-check file length + lint**

Run: `uv run python scripts/max_file_length.py && uv run ruff check aai_cli/agent_cascade/sandbox.py`
Expected: no output / clean. If `sandbox.py` is near 500 lines, it isn't — it should be ~220.

- [ ] **Step 6: Commit**

```bash
git add aai_cli/agent_cascade/sandbox.py tests/test_agent_cascade_sandbox.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): SandboxedShellBackend.execute confines to cwd or refuses"
```

---

### Task 5: Wire the backend, the gate, and memory into `brain.py`

Swap the backend, add `execute` to the approval set, turn on `MemoryMiddleware`, add the tool label and capability phrasing, and fix the stale "inert" comments.

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (`_build_fs_backend` ~199–208; `_WRITE_TOOLS` ~196; `_graph_kwargs` ~211–228; `_TOOL_LABELS` ~52–60; the `execute … inert` comment ~193–195)
- Modify: `aai_cli/agent_cascade/prompt.py` (capability phrasing ~27, ~52–71)
- Test: `tests/test_agent_cascade_brain.py`, `tests/test_agent_cascade_prompt.py`

**Interfaces:**
- Consumes: `sandbox.SandboxedShellBackend` (Task 4).
- Produces: `_build_fs_backend()` now returns a `SandboxedShellBackend`; `_WRITE_TOOLS == ("write_file", "edit_file", "execute")`; `_graph_kwargs(config)` (when `config.files`) additionally carries `memory=["./.deepagents/AGENTS.md"]`; `_TOOL_LABELS["execute"] == "Running code"`; the system prompt advertises code execution when `execute` is bound.

- [ ] **Step 1: Write the failing tests (brain)**

In `tests/test_agent_cascade_brain.py`, update the existing `test_graph_kwargs_*` test and add new ones. The existing assertion `kwargs["interrupt_on"] == {"write_file": True, "edit_file": True}` MUST change to include `execute` (this is the mutation-killing edit on the `_WRITE_TOOLS` line):

```python
def test_graph_kwargs_gates_writes_and_execute_and_sets_memory(monkeypatch, tmp_path):
    from aai_cli.agent_cascade import sandbox

    monkeypatch.chdir(tmp_path)
    kwargs = brain._graph_kwargs(CascadeConfig(files=True))

    backend = kwargs["backend"]
    assert isinstance(backend, sandbox.SandboxedShellBackend)
    assert Path(backend.cwd) == tmp_path.resolve()
    assert backend.virtual_mode is True
    # execute now joins the write gate.
    assert kwargs["interrupt_on"] == {"write_file": True, "edit_file": True, "execute": True}
    assert kwargs["checkpointer"] is not None
    # Durable per-project memory is turned on.
    assert kwargs["memory"] == ["./.deepagents/AGENTS.md"]


def test_graph_kwargs_empty_when_files_off():
    assert brain._graph_kwargs(CascadeConfig(files=False)) == {}


def test_sandboxed_backend_implements_sandbox_protocol(monkeypatch, tmp_path):
    from deepagents.backends.protocol import SandboxBackendProtocol

    monkeypatch.chdir(tmp_path)
    backend = brain._build_fs_backend()
    assert isinstance(backend, SandboxBackendProtocol)


def test_tool_label_execute_is_running_code():
    assert brain._tool_label("execute") == "Running code"
```

Also add a gated-decline test mirroring the existing write-decline coverage (find the test that drives `_stream_gated`/`_decide` with a rejecting approver and assert an `execute` action declines to `_DECLINED`). If the existing files-test (`tests/test_agent_cascade_files.py`) parametrizes the gated tool name, add `"execute"` to that parametrization; otherwise add:

```python
def test_declined_execute_yields_declined_message():
    action = {"name": "execute", "args": {"command": "rm -rf build"}}
    assert brain._decide(action, lambda name, args: False) == {
        "type": "reject",
        "message": brain._DECLINED,
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q -k "graph_kwargs or sandbox or tool_label or declined_execute"`
Expected: FAIL.

- [ ] **Step 3: Implement the brain edits**

In `brain.py`:

Replace the `_WRITE_TOOLS` block and its comment (lines ~193–196):

```python
# The mutating tools gated behind human approval when --files is on (reads — incl. grep — stay
# ungated). execute joins the gate because the backend is now sandbox-capable: it runs real
# commands in cwd, OS-confined, but every run is still approved.
_WRITE_TOOLS = ("write_file", "edit_file", "execute")
```

Replace `_build_fs_backend` (lines ~199–208):

```python
def _build_fs_backend() -> object:
    """A sandbox-capable deepagents backend rooted at the launch directory.

    ``virtual_mode=True`` maps the model's ``/``-rooted paths under cwd and blocks traversal
    escapes (same containment as before for file ops). Being a ``SandboxBackendProtocol`` backend
    is what makes deepagents bind a *functional* ``execute`` — and :class:`SandboxedShellBackend`
    runs it OS-sandboxed in cwd (no network, no escape) rather than on the host shell."""
    from aai_cli.agent_cascade.sandbox import SandboxedShellBackend

    return SandboxedShellBackend(root_dir=str(Path.cwd()), virtual_mode=True)
```

In `_graph_kwargs` (lines ~211–228) add the `memory` key to the returned dict:

```python
    return {
        "backend": backend_factory(),
        "interrupt_on": dict.fromkeys(_WRITE_TOOLS, True),
        "checkpointer": InMemorySaver(),
        "memory": ["./.deepagents/AGENTS.md"],
    }
```

In `_TOOL_LABELS` (lines ~52–60) add the execute label (keep the dict's existing entries):

```python
    "execute": "Running code",
```

- [ ] **Step 4: Implement the prompt edit**

In `prompt.py`, the file capability currently reads (line ~27):

```python
_FILE_CAPABILITY = "read, write, and search files in your working directory"
```

When `--files` is on, `execute` is bound, so the agent can run code. Update the phrasing so it advertises execution. Change `_FILE_CAPABILITY` to:

```python
_FILE_CAPABILITY = (
    "read, write, and search files in your working directory, and run code to solve problems "
    "and operate on this project"
)
```

(Single capability phrase; no new branch needed since `--files` is exactly when both the file tools and `execute` are bound. This keeps the change minimal and matches the spec's "advertises *run code…* when `execute` is bound.")

- [ ] **Step 5: Write/adjust the prompt test**

In `tests/test_agent_cascade_prompt.py`, find the test asserting the file-capability phrase appears when `files=True` and tighten it to assert the run-code phrasing (kills the mutation on the changed string — help/docstrings are snapshot-pinned, but `_FILE_CAPABILITY` is asserted directly here):

```python
def test_system_prompt_advertises_code_execution_under_files():
    prompt = build_system_prompt("persona", tools=[], files=True)
    assert "run code to solve problems" in prompt


def test_system_prompt_omits_code_execution_without_files():
    prompt = build_system_prompt("persona", tools=[], files=False)
    assert "run code" not in prompt
```

- [ ] **Step 6: Run to verify all pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py tests/test_agent_cascade_prompt.py tests/test_agent_cascade_files.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add aai_cli/agent_cascade/brain.py aai_cli/agent_cascade/prompt.py tests/test_agent_cascade_brain.py tests/test_agent_cascade_prompt.py tests/test_agent_cascade_files.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): sandbox-capable backend, gated execute, durable memory"
```

---

### Task 6: Make `risk.py`'s `execute` branch live + assert it

`risk.py`'s `execute` shell-risk warning was dormant (the live agent never bound a functional `execute`). It now surfaces on the approval prompt. The branch already exists; this task pins it with assertions so the now-live behavior can't regress (and the mutation gate on any touched line is satisfied).

**Files:**
- Test: `tests/test_agent_cascade_risk.py` (create if absent; otherwise extend the existing risk test file — search `tests/` for `risk_warning`)
- Modify (comment only, if present): `aai_cli/agent_cascade/risk.py` docstring noting the branch is live.

**Interfaces:**
- Consumes: `risk.risk_warning(name, args)` (existing).

- [ ] **Step 1: Write the tests**

```python
from __future__ import annotations

from aai_cli.agent_cascade import risk


def test_execute_warns_on_destructive_command():
    assert risk.risk_warning("execute", {"command": "rm -rf build"}) is not None
    assert risk.risk_warning("execute", {"command": "sudo make install"}) is not None


def test_execute_no_warning_on_benign_command():
    assert risk.risk_warning("execute", {"command": "pytest -q"}) is None


def test_execute_no_warning_when_command_missing_or_nonstring():
    assert risk.risk_warning("execute", {}) is None
    assert risk.risk_warning("execute", {"command": 123}) is None
```

- [ ] **Step 2: Run to verify they pass (branch already exists)**

Run: `uv run pytest tests/test_agent_cascade_risk.py -q`
Expected: PASS (the logic exists; these assertions make it *gate-enforced*).

> If the file already exists with these exact assertions, skip — `risk.py` was always tested; only confirm coverage. If `risk.py` needs no code change, there's nothing for the mutation gate to scope here.

- [ ] **Step 3: Commit (only if files changed)**

```bash
git add tests/test_agent_cascade_risk.py aai_cli/agent_cascade/risk.py
AAI_ALLOW_COMMIT=1 git commit -m "test(live): pin risk.py execute branch now that execute is gated"
```

---

### Task 7: Docs, help string + snapshot, and the full gate

Update the stale prose, the `--files` help string (regenerating its golden), and run the authoritative gate end-to-end.

**Files:**
- Modify: `aai_cli/AGENTS.md` (the `--files` paragraph, ~line 154)
- Modify: `aai_cli/commands/agent_cascade/__init__.py` (`--files` help string, ~174–179)
- Modify: `REFERENCE.md` (the `--files` description, ~163–167); `README.md` only if its `--files` blurb needs it
- Regenerate: `tests/__snapshots__/test_snapshots_help_run.ambr`

- [ ] **Step 1: Update the `--files` help string**

In `aai_cli/commands/agent_cascade/__init__.py`, change the `help=` to reflect code execution + memory (terse, no trailing period):

```python
        help="Let the agent read, write, and run code in the current directory, sandboxed (writes and runs need confirmation)",
```

- [ ] **Step 2: Regenerate the affected help snapshot**

Run: `uv run pytest tests/test_snapshots_help_run.py --snapshot-update -q`
Then eyeball the diff: `git diff tests/__snapshots__/test_snapshots_help_run.ambr` — only the `--files` line should change.

- [ ] **Step 3: Update `aai_cli/AGENTS.md`**

Replace the `--files` paragraph so it no longer says `execute` is inert. New text (keep it one paragraph, factual):

```
**`--files`** (off by default) swaps the brain's in-memory backend for a real-cwd, sandbox-capable
`SandboxedShellBackend` (`aai_cli/agent_cascade/sandbox.py`): file ops behave as before
(traversal-blocked `virtual_mode`), and because it implements `SandboxBackendProtocol` deepagents
binds a *functional* `execute` that runs commands OS-sandboxed in cwd — `sandbox-exec` (SBPL) on
macOS, `bwrap` on Linux, refused on any other platform/missing binary, never an unconfined
fallback (no network, writes confined to cwd, credential stores read-denied). `write_file`/
`edit_file`/`execute` are gated via `interrupt_on` + an `InMemorySaver`; `brain._stream_gated`
detects the post-stream interrupt, asks an injected `Approver`, and resumes with `Command(resume=…)`,
bracketing the human wait in `ApprovalPause` events so `engine._consume` suspends its reply
deadline. The voice TUI supplies the approver via `modals.ApprovalScreen` (`y`/`a`/`n`); headless
runs auto-deny (`_exec._deny_writes`). `--files` also turns on durable per-project memory via
`MemoryMiddleware` (`memory=["./.deepagents/AGENTS.md"]`). Reads (incl. `grep`) stay ungated.
```

- [ ] **Step 4: Update `REFERENCE.md`**

Update the `--files` description (~163–167) to mention sandboxed code execution + per-project memory, matching the new help string. Keep the existing tone. Ensure the docs-consistency gate stays green (no new env var/command is introduced, so the gate only checks the `--files` command reference still resolves).

- [ ] **Step 5: Run the full authoritative gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` Address anything it flags — likely candidates:
- patch-coverage < 100% → add the missing assertion for the uncovered changed line.
- a surviving mutant → strengthen the test so it *fails* when that line breaks.
- docstring-coverage → every public function in `sandbox.py` already has a docstring; add any missing.
- file-length → `sandbox.py` must be < 500 lines.
- docs-consistency → `REFERENCE.md`/`README.md` `--files` refs in sync.

- [ ] **Step 6: Final commit (gated)**

Once `check.sh` prints `All checks passed.`:

```bash
git add aai_cli/AGENTS.md aai_cli/commands/agent_cascade/__init__.py REFERENCE.md README.md tests/__snapshots__/test_snapshots_help_run.ambr
git commit -m "feat(live): document sandboxed execute + memory; refresh --files help"
```

(No `AAI_ALLOW_COMMIT=1` — the gate just passed, so the commit hook is satisfied.)

---

## Self-Review

**Spec coverage (M1 only):**
- Sandboxed gated `execute` in real cwd → Tasks 1–5. ✅
- OS sandbox, refuse-don't-fall-back → Task 3 (capability) + Task 4 (`none` branch, never-`super` test). ✅
- cwd-scoped reads default-allow + secrets read-denylist → Tasks 1–2 (renderers) + parity test. ✅
- writes confined to cwd + persistence write-denylist → Tasks 1–2. ✅
- no network → asserted in both renderer tests. ✅
- `execute` joins `interrupt_on`, flows through existing approver, `risk.py` warning live → Task 5 + Task 6. ✅
- durable memory via `MemoryMiddleware` (`memory=["./.deepagents/AGENTS.md"]`) → Task 5. ✅
- `_TOOL_LABELS["execute"]`, capability phrasing, stale-comment fixes, help/docs → Tasks 5 + 7. ✅
- no new dependency, live-only, never-raise contract → Global Constraints + Task 4 error handling. ✅
- **Deferred to later PRs (correctly out of M1):** subagents/`task` tool + HITL spike (M2); spoken approval + engine STT race + destructive-tier keyboard fallback (M3). The `_TOOL_LABELS["task"]` and the `task` capability phrase land in M2.

**Placeholder scan:** No TBD/"handle edge cases"/"similar to" — every code step shows the code. The two implementation notes (the `# noqa` removal in Tasks 3/4) are explicit instructions, not placeholders.

**Type consistency:** `Runner`, `Capability`, `CompletedProcessLike`, `_Result`, `render_seatbelt_profile`, `build_bwrap_argv`, `detect_capability`, `default_runner`, `SandboxedShellBackend`, `ExecuteResponse` are used with identical names/signatures across Tasks 1–5. `ExecuteResponse(output=…, exit_code=…)` matches deepagents' dataclass (`output: str`, `exit_code: int | None = None`).

## Execution Handoff

Open question for the implementer to confirm during Task 4: the exact `LocalShellBackend.__init__` keyword set (the explore pass found `FilesystemBackend.__init__(root_dir, virtual_mode, max_file_size_mb)` and `LocalShellBackend(FilesystemBackend, SandboxBackendProtocol)`) — if `LocalShellBackend.__init__` adds required kwargs, forward them. The injected-`runner` tests don't exercise the real binary, so CI (which has neither sandbox) stays green.
