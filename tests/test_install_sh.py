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
    #   -V          → print a version (used in the <3.12 error message)
    #   -c '<gate>' → exit 0/1 to pass/fail the 3.12+ gate
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
    assert "Python 3.12+ is required" in result.stderr


def test_errors_when_python_too_old(tmp_path):
    _python_shim(tmp_path, version="3.9.18", gate_ok=False)
    result = _run(tmp_path)
    assert result.returncode == 1
    assert "Python 3.12+ is required" in result.stderr
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
        tmp_path / "pip.args"
    ).read_text().strip() == f"pip install --user --upgrade {DEFAULT_SPEC}"


def test_repo_and_ref_override_the_spec(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    result = _run(tmp_path, {"AAI_REPO": "me/fork", "AAI_REF": "dev"})
    assert result.returncode == 0
    assert (
        tmp_path / "pipx.args"
    ).read_text().strip() == "install --force git+https://github.com/me/fork.git@dev"


def test_aai_spec_is_used_verbatim(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    result = _run(tmp_path, {"AAI_SPEC": "/tmp/aai_cli-0.1.0-py3-none-any.whl"})
    assert result.returncode == 0
    assert (
        tmp_path / "pipx.args"
    ).read_text().strip() == "install --force /tmp/aai_cli-0.1.0-py3-none-any.whl"


def test_path_hint_when_aai_not_on_path(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    result = _run(tmp_path)  # no `aai` shim → `command -v aai` fails
    assert result.returncode == 0
    assert "isn't on your PATH yet" in result.stdout


def test_next_steps_when_aai_present(tmp_path):
    _python_shim(tmp_path)
    _pipx_shim(tmp_path)
    _shim(tmp_path / "aai", "exit 0\n")
    result = _run(tmp_path)
    assert "Installed. Next: run 'aai onboard'" in result.stdout
