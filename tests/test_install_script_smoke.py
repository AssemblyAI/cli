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
