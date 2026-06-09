from pathlib import Path

import pytest

from aai_cli.errors import CLIError
from aai_cli.init import procfile

WEB = "web: uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-3000}\n"


def _write(tmp_path: Path, text: str) -> Path:
    (tmp_path / "Procfile").write_text(text)
    return tmp_path


def test_web_argv_expands_port_when_set(tmp_path):
    argv = procfile.web_argv(_write(tmp_path, WEB), env={"PORT": "8123"})
    assert argv[:2] == ["uvicorn", "api.index:app"]
    assert "--host" in argv
    assert argv[-2:] == ["--port", "8123"]


def test_web_argv_uses_default_when_port_unset(tmp_path):
    argv = procfile.web_argv(_write(tmp_path, WEB), env={})
    assert argv[-2:] == ["--port", "3000"]


def test_web_argv_default_when_var_is_empty(tmp_path):
    argv = procfile.web_argv(_write(tmp_path, WEB), env={"PORT": ""})
    assert argv[-2:] == ["--port", "3000"]


def test_web_argv_expands_plain_and_braced_vars(tmp_path):
    text = "web: run $HOST ${EXTRA}\n"
    argv = procfile.web_argv(_write(tmp_path, text), env={"HOST": "h", "EXTRA": "x"})
    assert argv == ["run", "h", "x"]


def test_web_argv_missing_var_becomes_empty(tmp_path):
    argv = procfile.web_argv(_write(tmp_path, "web: run ${NOPE} $ALSONOPE\n"), env={})
    assert argv == ["run", "", ""]


def test_web_argv_raises_without_procfile(tmp_path):
    with pytest.raises(CLIError) as exc:
        procfile.web_argv(tmp_path, env={})
    assert exc.value.error_type == "usage_error"
    assert "aai init" in str(exc.value)


def test_web_argv_raises_without_web_line(tmp_path):
    with pytest.raises(CLIError) as exc:
        procfile.web_argv(_write(tmp_path, "release: echo hi\n"), env={})
    assert exc.value.error_type == "usage_error"
    assert exc.value.exit_code == 1


def test_web_argv_raises_on_empty_web_command(tmp_path):
    with pytest.raises(CLIError):
        procfile.web_argv(_write(tmp_path, "web:\n"), env={})
