"""Real install-and-run smoke tests for the documented install paths.

Builds a wheel from the checkout once, then installs it into hermetic locations
and asserts the resulting `aai` binary actually runs — exercising dependency
resolution + the console entrypoint for each documented installer:

* install.sh's pipx and pip --user branches (via the test-only AAI_SPEC
  override, which installs the wheel instead of the public git URL so tests
  need no push); install.sh's pipx branch is literally `pipx install <wheel>`,
  so it doubles as coverage of the README's `pipx install` command.
* `uv tool install <wheel>` — the README's uv path.

The Homebrew path is covered separately by the `formula-install` CI job, which
does a real `brew install` of the formula built against the branch source.

Marked `install_script`: slow + needs network (deps resolve from PyPI) and
pipx/uv. Excluded from the default run; invoke explicitly::

    uv run pytest -m install_script

CI runs every branch on Linux; on macOS it runs the pipx + uv tool branches
(`pip --user` is flaky there under PEP 668).
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
    wheels: list[Path] = list(out.glob("*.whl"))
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


def test_install_via_uv_tool(built_wheel: Path, tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; required for the uv tool install branch")
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable; skipping real-install smoke test (offline)")

    # Hermetic uv tool dirs so the install never touches the developer's real
    # toolchain; UV_TOOL_BIN_DIR is where uv links the `aai` entrypoint.
    bindir = tmp_path / "uv_bin"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tmp_path / "uv_tools"),
        "UV_TOOL_BIN_DIR": str(bindir),
    }
    run = subprocess.run(
        ["uv", "tool", "install", str(built_wheel)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    _assert_aai_runs(bindir / "aai")
