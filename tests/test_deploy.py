from __future__ import annotations

import dataclasses
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.deploy import RAILWAY, VERCEL, Target
from aai_cli.main import app

runner = CliRunner()


def test_targets_are_frozen() -> None:
    # The default and Railway targets are module-level singletons; freezing them
    # guards against accidental in-place mutation of shared deploy config.
    # Route the assignment through an object-typed alias and a runtime attribute
    # name so the frozen-ness is checked at runtime, not statically.
    field = "name"
    for target in (VERCEL, RAILWAY):
        opaque: object = target
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(opaque, field, "tampered")
    assert isinstance(VERCEL, Target)


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available: Sequence[str] = ("vercel",),
    agentic: bool = False,
    confirm: bool = True,
    returncode: int = 0,
) -> dict[str, object]:
    monkeypatch.setattr(
        "shutil.which", lambda name: f"/usr/bin/{name}" if name in available else None
    )
    monkeypatch.setattr("aai_cli.output.is_agentic", lambda: agentic)
    calls: dict[str, object] = {}

    def fake_confirm(prompt: str, *a: object, **k: object) -> bool:
        calls["prompt"] = prompt
        return confirm

    monkeypatch.setattr("typer.confirm", fake_confirm)

    def fake_run(cmd: list[str], **kwargs: object) -> types.SimpleNamespace:
        calls["cmd"] = cmd
        calls["cwd"] = kwargs.get("cwd")
        calls["check"] = kwargs.get("check")
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.deploy.subprocess.run", fake_run)
    return calls


def test_deploy_defaults_to_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy"]
    assert calls["check"] is False
    assert calls["cwd"] == Path.cwd()
    assert calls["prompt"] == "Deploy this project to Vercel?"


def test_deploy_vercel_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--vercel", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy"]


def test_deploy_railway_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["railway", "up"]


def test_deploy_railway_prompt_names_railway(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",), confirm=False)
    result = runner.invoke(app, ["deploy", "--railway"])
    assert result.exit_code == 0, result.output
    assert calls["prompt"] == "Deploy this project to Railway?"
    assert "cmd" not in calls  # declined


def test_deploy_both_targets_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel", "railway"))
    result = runner.invoke(app, ["deploy", "--vercel", "--railway", "--yes"])
    assert result.exit_code == 1
    assert "not both" in result.output
    assert "cmd" not in calls  # never deployed


def test_deploy_missing_vercel_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 1
    assert "Vercel CLI" in result.output
    assert "npm i -g vercel" in " ".join(result.output.split())


def test_deploy_missing_railway_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--railway", "--yes"])
    assert result.exit_code == 1
    assert "Railway CLI" in result.output
    # Console may soft-wrap the hint, so normalize whitespace before matching.
    assert "npm i -g @railway/cli" in " ".join(result.output.split())


def test_deploy_confirm_no_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), confirm=False)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output
    assert "cmd" not in calls


def test_deploy_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), confirm=False)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy"]
    assert "prompt" not in calls  # --yes bypassed typer.confirm


def test_deploy_prod_flag_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--prod", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["vercel", "deploy", "--prod"]


def test_deploy_prod_ignored_for_railway(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--prod", "--yes"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["railway", "up"]


def test_deploy_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=("vercel",), returncode=2)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 2


def test_deploy_noninteractive_without_yes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), agentic=True)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert "cmd" not in calls


@pytest.mark.parametrize("flag", ["--prod", "--vercel", "--railway", "--yes"])
def test_deploy_help_lists_flags(flag: str) -> None:
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert flag in result.output
