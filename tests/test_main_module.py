import subprocess
import sys


def test_python_dash_m_entrypoint_runs():
    """`python -m assemblyai_cli` wires up the Typer app (exercises __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "assemblyai_cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "aai" in result.stdout


def test_python_dash_m_version():
    result = subprocess.run(
        [sys.executable, "-m", "assemblyai_cli", "version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip()  # prints something (the version)
