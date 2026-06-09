from __future__ import annotations

import dataclasses
import re
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.deploy import FLY, RAILWAY, VERCEL, Target
from aai_cli.main import app

runner = CliRunner()

# CI forces color; Rich then styles option flags with ANSI codes inserted mid-token
# (e.g. `--<ESC>[…m-fly`), so the literal "--fly" isn't a substring. Strip ANSI first.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_targets_are_frozen() -> None:
    # Every deploy target is a module-level singleton; freezing them guards
    # against accidental in-place mutation of shared deploy config.
    # Route the assignment through an object-typed alias and a runtime attribute
    # name so the frozen-ness is checked at runtime, not statically.
    field = "name"
    for target in (VERCEL, RAILWAY, FLY):
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
        runs = calls.setdefault("runs", [])
        assert isinstance(runs, list)
        runs.append({"cmd": cmd, "cwd": kwargs.get("cwd"), "check": kwargs.get("check")})
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.deploy.subprocess.run", fake_run)
    return calls


def _runs(calls: dict[str, object]) -> list[dict[str, object]]:
    """Every captured subprocess.run call, in order."""
    runs = calls.get("runs", [])
    assert isinstance(runs, list)
    out: list[dict[str, object]] = []
    for run in runs:
        assert isinstance(run, dict)
        out.append(run)
    return out


def _cmds(calls: dict[str, object]) -> list[list[str]]:
    """The command argv for every captured subprocess.run, in call order."""
    out: list[list[str]] = []
    for run in _runs(calls):
        cmd = run["cmd"]
        assert isinstance(cmd, list)
        out.append(cmd)
    return out


def test_deploy_defaults_to_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 0, result.output
    run = _runs(calls)[0]
    assert run["cmd"] == ["vercel", "deploy"]
    assert run["check"] is False
    assert run["cwd"] == Path.cwd()
    assert calls["prompt"] == "Deploy this project to Vercel?"


def test_deploy_vercel_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--vercel", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls) == [["vercel", "deploy"]]


def test_deploy_vercel_no_post_deploy(monkeypatch: pytest.MonkeyPatch) -> None:
    # Vercel has no post_deploy_args, so exactly one subprocess.run fires.
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--vercel", "--yes"])
    assert result.exit_code == 0, result.output
    assert len(_runs(calls)) == 1
    assert _runs(calls)[0]["cmd"] == ["vercel", "deploy"]


def test_deploy_railway_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls)[0] == ["railway", "up"]


def test_deploy_railway_generates_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    # A successful railway deploy chases the deploy with `railway domain` so the
    # public URL is surfaced (railway up alone prints no URL).
    calls = _stub(monkeypatch, available=("railway",), returncode=0)
    result = runner.invoke(app, ["deploy", "--railway", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls) == [["railway", "up"], ["railway", "domain"]]
    # The post-deploy step is best-effort: its non-zero exit must not propagate,
    # so it too runs with check=False.
    assert _runs(calls)[1]["check"] is False


def test_deploy_railway_failed_skips_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    # A non-zero deploy raises before the post-deploy step, so `railway domain`
    # never runs and the deploy's exit code propagates.
    calls = _stub(monkeypatch, available=("railway",), returncode=2)
    result = runner.invoke(app, ["deploy", "--railway", "--yes"])
    assert result.exit_code == 2
    assert _cmds(calls) == [["railway", "up"]]


def test_deploy_railway_prompt_names_railway(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",), confirm=False)
    result = runner.invoke(app, ["deploy", "--railway"])
    assert result.exit_code == 0, result.output
    assert calls["prompt"] == "Deploy this project to Railway?"
    assert _cmds(calls) == []  # declined


def test_deploy_multiple_targets_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel", "railway"))
    result = runner.invoke(app, ["deploy", "--vercel", "--railway", "--yes"])
    assert result.exit_code == 1
    assert "at most one" in result.output
    # The error lists every target flag, including the new ones.
    assert "--fly" in result.output
    assert "--render" not in result.output
    assert _cmds(calls) == []  # never deployed


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
    assert _cmds(calls) == []


def test_deploy_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), confirm=False)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls)[0] == ["vercel", "deploy"]
    assert "prompt" not in calls  # --yes bypassed typer.confirm


def test_deploy_prod_flag_vercel(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--prod", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls)[0] == ["vercel", "deploy", "--prod"]


def test_deploy_prod_ignored_for_railway(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--prod", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls)[0] == ["railway", "up"]


def test_deploy_fly_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # `fly launch` creates the app, generates fly.toml (detecting the shipped
    # Dockerfile), and deploys — one command, no fly.toml needed beforehand.
    calls = _stub(monkeypatch, available=("fly",))
    result = runner.invoke(app, ["deploy", "--fly", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls) == [["fly", "launch"]]


def test_deploy_prod_ignored_for_fly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("fly",))
    result = runner.invoke(app, ["deploy", "--fly", "--prod", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls)[0] == ["fly", "launch"]


def test_deploy_missing_fly_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--fly", "--yes"])
    assert result.exit_code == 1
    assert "Fly CLI" in result.output
    assert "brew install flyctl" in " ".join(result.output.split())


def test_deploy_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=("vercel",), returncode=2)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 2


def test_deploy_noninteractive_without_yes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), agentic=True)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 1
    assert "--yes" in result.output
    assert _cmds(calls) == []


@pytest.mark.parametrize("flag", ["--prod", "--vercel", "--railway", "--fly", "--yes"])
def test_deploy_help_lists_flags(flag: str) -> None:
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert flag in _ANSI.sub("", result.output)
