from __future__ import annotations

import types
from typing import Any

import pytest
from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    has_vercel: bool = True,
    agentic: bool = False,
    confirm: bool = True,
    returncode: int = 0,
) -> dict[str, Any]:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/vercel" if has_vercel else None)
    monkeypatch.setattr("aai_cli.output.is_agentic", lambda: agentic)
    monkeypatch.setattr("typer.confirm", lambda *a, **k: confirm)
    calls: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> types.SimpleNamespace:
        calls["cmd"] = cmd
        calls["cwd"] = kwargs.get("cwd")
        calls["check"] = kwargs.get("check")
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.deploy.subprocess.run", fake_run)
    return calls


def test_deploy_missing_vercel_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, has_vercel=False)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 1
    assert "npm i -g vercel" in result.output


def test_deploy_confirm_no_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, agentic=False, confirm=False)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output
    assert calls == {}  # vercel never invoked


def test_deploy_confirm_yes_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, agentic=False, confirm=True)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy"]


def test_deploy_yes_flag_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    # confirm=False would abort if the prompt were consulted; --yes must bypass it.
    calls = _stub(monkeypatch, agentic=False, confirm=False)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy"]


def test_deploy_prod_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, confirm=True)
    result = runner.invoke(app, ["deploy", "--yes", "--prod"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy", "--prod"]


def test_deploy_runs_in_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, confirm=True)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 0, result.output
    from pathlib import Path

    assert calls["cwd"] == Path.cwd()
    # We handle the exit code ourselves; subprocess must not raise on failure.
    assert calls["check"] is False


def test_deploy_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, confirm=True, returncode=2)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 2


def test_deploy_noninteractive_without_yes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, agentic=True, confirm=True)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert calls == {}  # never deployed


@pytest.mark.parametrize("flag", ["--prod", "--yes"])
def test_deploy_help_lists_flags(flag: str) -> None:
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert flag in result.output
