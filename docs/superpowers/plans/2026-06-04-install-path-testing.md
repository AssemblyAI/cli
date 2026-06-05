# Install-path Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test the public install path (`install.sh` → pipx/pip → working `aai`) in layers, so a broken entrypoint, dependency floor, or `install.sh` bug fails CI instead of shipping.

**Architecture:** Three layers. (1) Fast shell-logic unit tests that run `install.sh` under a sandboxed PATH of fake shims — no network, in the default suite. (2) A small `AAI_SPEC` testability seam in `install.sh` so tests install the PR's own code from a locally-built wheel without pushing. (3) A real install-and-run smoke test (marked `install_script`) that builds a wheel, runs `install.sh` against it hermetically, and asserts `aai version` works — exercised by a new `install-smoke` CI job (both pipx/pip branches on Linux, pipx-only on macOS).

**Tech Stack:** POSIX `sh` (`install.sh`), Python + pytest + `subprocess`, `uv build`, pipx, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-06-04-install-path-testing-design.md`

**Note on commits:** `docs/` is in `.gitignore`, so the spec and this plan are untracked on purpose. Every commit below adds only real source files (never the plan/spec).

---

### Task 1: Shell-logic unit tests + `AAI_SPEC` seam

This is one red→green cycle: the test file asserts current behavior (passes immediately) **plus** the not-yet-existing `AAI_SPEC` override (fails). Step 3 adds the seam to make it green.

**Files:**
- Create: `tests/test_install_sh.py`
- Modify: `install.sh:9-11`

- [ ] **Step 1: Write the shell-logic test file**

```python
"""Shell-logic unit tests for install.sh.

Run install.sh under a sandboxed PATH of fake shims so we can assert *which*
installer it invokes and with *what* spec — without any real install or network.
The shims record their argv to files in a temp dir; the script's only external
dependencies (python3, pipx, and pip via `python -m pip`) are all faked.

Fast; runs in the default suite. The real install-and-boot test lives in
test_install_script_smoke.py (marked `install_script`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parent.parent / "install.sh"
DEFAULT_SPEC = "git+https://github.com/AssemblyAI/cli.git@main"


def _sh() -> str:
    return shutil.which("sh") or "/bin/sh"


def _shim(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _python_shim(bindir: Path, *, version: str = "3.12.0", gate_ok: bool = True) -> None:
    # Fakes the three ways install.sh calls python:
    #   -V          → print a version (used in the <3.10 error message)
    #   -c '<gate>' → exit 0/1 to pass/fail the 3.10+ gate
    #   -m pip ...  → record argv to pip.args (the pip --user fallback)
    rec = bindir / "pip.args"
    _shim(
        bindir / "python3",
        f'case "$1" in\n'
        f'  -V|--version) echo "Python {version}"; exit 0 ;;\n'
        f"  -c) exit {0 if gate_ok else 1} ;;\n"
        f'  -m) shift; echo "$@" > "{rec}"; exit 0 ;;\n'
        f"esac\n"
        f"exit 0\n",
    )


def _pipx_shim(bindir: Path) -> None:
    rec = bindir / "pipx.args"
    _shim(bindir / "pipx", f'echo "$@" > "{rec}"\nexit 0\n')


def _run(bindir: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": str(bindir)}
    if env_extra:
        env.update(env_extra)
    return subprocess.run([_sh(), str(INSTALL_SH)], env=env, capture_output=True, text=True)


def test_errors_when_no_python(tmp_path):
    # Empty bindir: no python3/python on PATH at all.
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "Python 3.10+ is required" in result.stderr


def test_errors_when_python_too_old(tmp_path):
    _python_shim(tmp_path, version="3.9.18", gate_ok=False)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "Python 3.10+ is required" in result.stderr
    assert "3.9.18" in result.stderr


def test_uses_pipx_when_present(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    result = _run(tmp_path)
    assert result.returncode == 0
    assert (tmp_path / "pipx.args").read_text().strip() == f"install --force {DEFAULT_SPEC}"
    assert not (tmp_path / "pip.args").exists()  # pip fallback not taken


def test_falls_back_to_pip_user_when_no_pipx(tmp_path):
    _python_shim(tmp_path)  # no pipx shim → `command -v pipx` fails
    result = _run(tmp_path)
    assert result.returncode == 0
    assert (
        (tmp_path / "pip.args").read_text().strip()
        == f"pip install --user --upgrade {DEFAULT_SPEC}"
    )


def test_repo_and_ref_override_the_spec(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    _run(tmp_path, {"AAI_REPO": "me/fork", "AAI_REF": "dev"})
    assert (
        (tmp_path / "pipx.args").read_text().strip()
        == "install --force git+https://github.com/me/fork.git@dev"
    )


def test_aai_spec_is_used_verbatim(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    _run(tmp_path, {"AAI_SPEC": "/tmp/aai_cli-0.1.0-py3-none-any.whl"})
    assert (
        (tmp_path / "pipx.args").read_text().strip()
        == "install --force /tmp/aai_cli-0.1.0-py3-none-any.whl"
    )


def test_path_hint_when_aai_not_on_path(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    result = _run(tmp_path)  # no `aai` shim → `command -v aai` fails
    assert "isn't on your PATH yet" in result.stdout


def test_next_steps_when_aai_present(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    _shim(tmp_path / "aai", "exit 0\n")
    result = _run(tmp_path)
    assert "Installed. Next: run 'aai login'" in result.stdout
```

- [ ] **Step 2: Run the tests to confirm only the AAI_SPEC case fails**

Run: `uv run pytest tests/test_install_sh.py -v`
Expected: 7 PASS, `test_aai_spec_is_used_verbatim` FAILS — the assertion sees the default git spec because `install.sh` ignores `AAI_SPEC` today.

- [ ] **Step 3: Add the `AAI_SPEC` seam to install.sh**

In `install.sh`, replace lines 9-11:

```sh
REPO="${AAI_REPO:-AssemblyAI/cli}"
REF="${AAI_REF:-main}"
SPEC="git+https://github.com/${REPO}.git@${REF}"
```

with:

```sh
REPO="${AAI_REPO:-AssemblyAI/cli}"
REF="${AAI_REF:-main}"
# AAI_SPEC (test-only) installs an arbitrary pip spec verbatim — e.g. a locally
# built wheel — instead of the public git URL, so tests can exercise this script
# against the current checkout without pushing. Unset for normal installs.
SPEC="${AAI_SPEC:-git+https://github.com/${REPO}.git@${REF}}"
```

- [ ] **Step 4: Run the tests to confirm all pass**

Run: `uv run pytest tests/test_install_sh.py -v`
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_install_sh.py install.sh
git commit -m "test(install): shell-logic unit tests + AAI_SPEC seam for install.sh"
```

---

### Task 2: shellcheck gate in check.sh

**Files:**
- Modify: `scripts/check.sh` (add a step before the pytest step)

- [ ] **Step 1: Confirm install.sh is already clean**

Run: `shellcheck install.sh`
Expected: no output, exit 0. (If shellcheck reports anything, fix `install.sh` accordingly before continuing — these should be minor, e.g. quoting.)

- [ ] **Step 2: Add the shellcheck step to check.sh**

In `scripts/check.sh`, immediately **before** the `echo "==> pytest ..."` block, insert:

```bash
echo "==> shellcheck (install.sh)"
# Static-lint the public install script. CI's ubuntu runner ships shellcheck;
# locally it's skipped with a notice if not installed.
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck install.sh
else
  echo "   shellcheck not found; skipping (CI runs it)"
fi
```

- [ ] **Step 3: Run check.sh far enough to see shellcheck pass**

Run: `shellcheck install.sh && echo OK`
Expected: `OK`. (Running the full `./scripts/check.sh` also works but is slower; the targeted command verifies the new gate.)

- [ ] **Step 4: Commit**

```bash
git add scripts/check.sh
git commit -m "ci: shellcheck install.sh in check.sh"
```

---

### Task 3: Real install-and-run smoke test

**Files:**
- Create: `tests/test_install_script_smoke.py`
- Modify: `pyproject.toml:81-84` (register the `install_script` marker)
- Modify: `scripts/check.sh` (exclude `install_script` from the default pytest run)

- [ ] **Step 1: Register the `install_script` marker**

In `pyproject.toml`, in the `[tool.pytest.ini_options]` `markers = [...]` list (currently ending after the `install:` entry on line 83), add a third entry:

```toml
    "install_script: real install via install.sh from a locally-built wheel; asserts `aai` runs (slow; needs network + uv/pipx; skip otherwise)",
```

The list should now read:

```toml
markers = [
    "e2e: real-API end-to-end tests that drive the CLI (need ASSEMBLYAI_API_KEY + kokoro; skip otherwise)",
    "install: install each init template's requirements.txt into a clean venv and import it (slow; needs network + uv; skip otherwise)",
    "install_script: real install via install.sh from a locally-built wheel; asserts `aai` runs (slow; needs network + uv/pipx; skip otherwise)",
]
```

- [ ] **Step 2: Exclude the marker from the default pytest run**

In `scripts/check.sh`, change the pytest invocation's marker filter from:

```bash
uv run pytest -q -m "not e2e and not install" --cov=aai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90
```

to:

```bash
uv run pytest -q -m "not e2e and not install and not install_script" --cov=aai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90
```

Also update the comment just above it to list the new marker alongside the others:

```bash
# Exclude e2e (live API key + kokoro), install (per-template dep install), and
# install_script (real install via install.sh). All are slow/networked and
# uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
#   uv run pytest -m install_script
```

- [ ] **Step 3: Write the smoke test**

Create `tests/test_install_script_smoke.py`:

```python
"""Real install-and-run smoke test for install.sh.

Builds a wheel from the checkout and runs install.sh against it (via the
test-only AAI_SPEC override) into a hermetic location, then asserts the
installed `aai` binary actually runs. This is the one check that exercises the
public install path end to end: dependency resolution, the console entrypoint,
and the pipx / pip --user branches in install.sh.

Marked `install_script`: slow + needs network (deps resolve from PyPI) and
pipx/uv. Excluded from the default run; invoke explicitly::

    uv run pytest -m install_script

The two tests map to the two install branches; CI runs both on Linux and only
the pipx branch on macOS.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from aai_cli import __version__

pytestmark = pytest.mark.install_script

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "install.sh"


@functools.lru_cache(maxsize=1)
def _pypi_reachable() -> bool:
    # Cached: both tests ask the same question, so probe the network once.
    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _sh() -> str:
    return shutil.which("sh") or "/bin/sh"


@pytest.fixture(scope="session")
def built_wheel(tmp_path_factory) -> Path:
    # Skip (never fail) when the machine can't build the wheel — mirrors the
    # template install test, so offline/sandboxed local runs aren't blocked.
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; needed to build the wheel under test")
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def _assert_aai_runs(aai_bin: Path) -> None:
    assert aai_bin.is_file(), f"install.sh did not produce {aai_bin}"
    result = subprocess.run([str(aai_bin), "version"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__


def test_install_via_pipx(built_wheel: Path, tmp_path: Path) -> None:
    if shutil.which("pipx") is None:
        pytest.skip("pipx not on PATH; required for the pipx install branch")
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable; skipping real-install smoke test (offline)")

    pipx_bin = tmp_path / "pipx_bin"
    # Inherit the real env so pipx/python resolve normally; the overrides keep
    # the install hermetic (its own pipx home + an isolated bin dir).
    env = {
        **os.environ,
        "AAI_SPEC": str(built_wheel),
        "PIPX_HOME": str(tmp_path / "pipx_home"),
        "PIPX_BIN_DIR": str(pipx_bin),
    }
    run = subprocess.run([_sh(), str(INSTALL_SH)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    _assert_aai_runs(pipx_bin / "aai")


def test_install_via_pip_user(built_wheel: Path, tmp_path: Path) -> None:
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable; skipping real-install smoke test (offline)")

    # Hermetic PATH with ONLY python3 → `command -v pipx` fails, forcing the
    # pip --user fallback. pip --user honors PYTHONUSERBASE for the install root.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = shutil.which("python3") or sys.executable
    (bindir / "python3").symlink_to(python)
    userbase = tmp_path / "userbase"
    env = {
        "PATH": str(bindir),
        "AAI_SPEC": str(built_wheel),
        "PYTHONUSERBASE": str(userbase),
    }
    run = subprocess.run([_sh(), str(INSTALL_SH)], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    _assert_aai_runs(userbase / "bin" / "aai")
```

- [ ] **Step 4: Run the smoke test (needs network + uv + pipx)**

Run: `uv run pytest -m install_script -v`
Expected: `test_install_via_pipx` and `test_install_via_pip_user` both PASS. (If `pip --user` is rejected because `uv run` supplies a virtualenv interpreter, run instead with a non-venv Python: `python -m pytest -m install_script -v` after `pip install -e ".[dev]" uv pipx`. CI uses exactly that non-venv invocation — see Task 4.)

- [ ] **Step 5: Confirm the default suite still excludes it**

Run: `uv run pytest -q -m "not e2e and not install and not install_script" --co | tail -3`
Expected: collection completes and lists no `test_install_script_smoke` items.

- [ ] **Step 6: Commit**

```bash
git add tests/test_install_script_smoke.py pyproject.toml scripts/check.sh
git commit -m "test(install): real install-and-run smoke test (marker: install_script)"
```

---

### Task 4: `install-smoke` CI job

**Files:**
- Modify: `.github/workflows/ci.yml` (add a new top-level job under `jobs:`)

- [ ] **Step 1: Add the `install-smoke` job**

In `.github/workflows/ci.yml`, append a new job to the `jobs:` map (after the `audit:` job). Reuse the same pinned action SHAs already used elsewhere in this file:

```yaml
  install-smoke:
    name: install.sh real install (${{ matrix.os }})
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            kfilter: ""             # both branches: pipx + pip --user
          - os: macos-latest
            kfilter: "-k pipx"      # pipx only — PEP 668 makes pip --user flaky on macOS
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6.0.3
      - uses: actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405 # v6.2.0
        with:
          python-version: "3.12"
          cache: pip

      # `aai version` imports the package, which pulls in sounddevice (needs
      # PortAudio) and ffmpeg-backed sources. Match the other jobs' system deps.
      - name: System deps (Linux)
        if: runner.os == 'Linux'
        run: sudo apt-get update && sudo apt-get install -y libportaudio2 ffmpeg
      - name: System deps (macOS)
        if: runner.os == 'macOS'
        run: brew install portaudio ffmpeg

      # Use the system interpreter (no virtualenv) so install.sh's `pip --user`
      # branch is allowed. Editable install makes `aai_cli` importable for the
      # test's __version__ check; uv builds the wheel; pipx drives the pipx branch.
      - name: Tooling
        run: python -m pip install -e ".[dev]" uv pipx

      - name: Real install smoke
        run: python -m pytest -q -m install_script ${{ matrix.kfilter }}
```

- [ ] **Step 2: Validate the workflow YAML**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok`. (If `actionlint` is installed, also run `actionlint .github/workflows/ci.yml` and expect no errors.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add install-smoke job exercising install.sh end to end"
```

- [ ] **Step 4: Push and confirm the job runs green**

```bash
git push
```
Expected: on the PR, the new `install.sh real install (ubuntu-latest)` and `(macos-latest)` checks run; ubuntu runs both branches, macOS runs only the pipx branch; all green. (This is the real cross-OS validation — local runs can't prove the macOS path.)

---

## Self-Review

**Spec coverage:**
- Layer 0 (`AAI_SPEC` seam) → Task 1, Step 3. ✅
- Layer 1 (fast shell-logic tests, in `check` job) → Task 1 (`tests/test_install_sh.py`, runs in default suite). ✅
- shellcheck static gate → Task 2. ✅
- Layer 2 (real install via marker `install_script`, skip-not-fail offline) → Task 3. ✅
- Layer 3 (`install-smoke` CI job, ubuntu both branches + macOS pipx-only) → Task 4. ✅
- Files-touched list in spec (install.sh, two test files, pyproject, check.sh, ci.yml) → all covered across Tasks 1–4. ✅

**Placeholder scan:** No TBD/TODO; every code step shows complete content; every command has an expected result. ✅

**Type/name consistency:** `DEFAULT_SPEC`, `_sh()`, `_shim()`, `_python_shim()`, `_pipx_shim()`, `_run()` used consistently in Task 1. `built_wheel` fixture and `_assert_aai_runs()` used consistently in Task 3. Marker name `install_script` identical across pyproject, check.sh, test `pytestmark`, and the CI `-m` filter. `AAI_SPEC` identical in install.sh and both test files. ✅

**Known risk surfaced inline:** `pip --user` is rejected inside a virtualenv — Task 3 Step 4 and Task 4 use a non-venv system interpreter to avoid this. macOS `pip --user`/PEP 668 deliberately excluded from the gating matrix (Task 4) per the spec.
