"""Install-and-boot smoke test: prove each template installs from its own
``requirements.txt`` into a clean environment and that ``api/index.py`` imports.

This is the one drift the static contract tests (``test_init_template_contract.py``)
can't catch: a dependency that's listed but uninstallable, a missing transitive
pin, or an import that only fails once the real packages are actually present.
The contract test checks every import *appears* in requirements.txt; this one
checks those requirements *resolve and run*.

Marked ``install``: it does a real dependency install per template, so it's slow
and needs network + ``uv``. It's excluded from the default run and the precommit
gate. Run it with::

    uv run pytest -m install
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.install

TEMPLATES_ROOT = Path("aai_cli/init/templates")
TEMPLATE_DIRS = sorted(
    p for p in TEMPLATES_ROOT.iterdir() if p.is_dir() and not p.name.startswith("__")
)


def _pypi_reachable() -> bool:
    try:
        urllib.request.urlopen("https://pypi.org/simple/", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


@pytest.mark.parametrize("template_dir", TEMPLATE_DIRS, ids=lambda p: p.name)
def test_template_installs_and_app_imports(template_dir: Path, tmp_path: Path) -> None:
    # Skip (never fail) when the machine can't run the test, mirroring the e2e
    # suite: keyless/offline contributors and sandboxed CI aren't blocked.
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH; the install test uses it to build the venv")
    if not _pypi_reachable():
        pytest.skip("PyPI unreachable; skipping install-and-boot test (offline)")

    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True, text=True)
    py = venv / ("Scripts" if sys.platform == "win32" else "bin") / "python"

    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(py), "-r", str(template_dir / "requirements.txt")],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, (
        f"{template_dir.name}: requirements.txt failed to install into a clean venv\n"
        f"{install.stderr}"
    )

    # Import api/index.py with ONLY its declared deps present — no key needed, the
    # module reads ASSEMBLYAI_API_KEY at import but defaults to "". A clean exit
    # proves the app boots from exactly what `aai init` ships to the user.
    boot = tmp_path / "boot.py"
    boot.write_text(
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('tmpl', sys.argv[1])\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert hasattr(mod, 'app'), 'template api/index.py does not export `app`'\n"
    )
    run = subprocess.run(
        [str(py), str(boot), str(template_dir / "api" / "index.py")],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, (
        f"{template_dir.name}: app failed to import with only its declared deps\n{run.stderr}"
    )
