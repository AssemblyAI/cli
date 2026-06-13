from __future__ import annotations

import dataclasses
import re
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.deploy._exec import FLY, RAILWAY, VERCEL, Target
from aai_cli.main import app

runner = CliRunner()

# CI forces color; Rich then styles option flags with ANSI codes inserted mid-token
# (e.g. `--<ESC>[…m-fly`), so the literal "--fly" isn't a substring. Strip ANSI first.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def in_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test inside a scaffolded-looking project (deploy guards on ./Procfile)."""
    monkeypatch.chdir(tmp_path)
    procfile = tmp_path / "Procfile"
    procfile.write_text("web: python -m uvicorn api.index:app --host 0.0.0.0 --port 3000\n")
    return procfile


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
    monkeypatch.setattr("aai_cli.ui.output.is_agentic", lambda: agentic)
    calls: dict[str, object] = {}

    def fake_confirm(prompt: str) -> bool:
        calls["prompt"] = prompt
        return confirm

    monkeypatch.setattr("typer.confirm", fake_confirm)

    def fake_run(cmd: list[str], *, cwd: Path, check: bool) -> types.SimpleNamespace:
        runs = calls.setdefault("runs", [])
        assert isinstance(runs, list)
        runs.append({"cmd": cmd, "cwd": cwd, "check": check})
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.deploy._exec.subprocess.run", fake_run)
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
    assert result.exit_code == 2  # usage error: the conventional exit 2
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
    # Human mode prints plain text, not the JSON shape.
    assert '"status"' not in result.output
    assert _cmds(calls) == []


def test_deploy_json_flag_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # deploy now has the standard --json flag like its init/dev/share siblings.
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    assert "No such option" not in result.output
    assert _cmds(calls) == [["vercel", "deploy"]]


def test_deploy_json_abort_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    calls = _stub(monkeypatch, available=("vercel",), confirm=False)
    result = runner.invoke(app, ["deploy", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "aborted", "target": "Vercel"}
    assert "Aborted." not in result.stdout
    assert _cmds(calls) == []


def test_deploy_json_error_is_enveloped(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--yes", "--json"])
    assert result.exit_code == 1
    err = json.loads(result.stderr)
    assert err["error"]["type"] == "missing_dependency"
    assert "Vercel CLI" in err["error"]["message"]


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


def test_deploy_prod_rejected_for_railway(monkeypatch: pytest.MonkeyPatch) -> None:
    # --prod only means something to Vercel; silently dropping it would deploy a
    # preview the user believed was production. Clean usage error instead.
    calls = _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--prod", "--yes"])
    assert result.exit_code == 2
    assert "--prod is only supported for Vercel deploys." in result.output
    assert _cmds(calls) == []  # nothing was deployed


def test_deploy_fly_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # `fly launch` creates the app, generates fly.toml (detecting the shipped
    # Dockerfile), and deploys — one command, no fly.toml needed beforehand.
    calls = _stub(monkeypatch, available=("fly",))
    result = runner.invoke(app, ["deploy", "--fly", "--yes"])
    assert result.exit_code == 0, result.output
    assert _cmds(calls) == [["fly", "launch"]]


def test_deploy_prod_rejected_for_fly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("fly",))
    result = runner.invoke(app, ["deploy", "--fly", "--prod", "--yes"])
    assert result.exit_code == 2
    assert "--prod is only supported for Vercel deploys." in result.output
    assert _cmds(calls) == []


def test_deploy_missing_fly_errors_with_brew_hint_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--fly", "--yes"])
    assert result.exit_code == 1
    assert "Fly CLI" in result.output
    assert "brew install flyctl" in " ".join(result.output.split())


def test_deploy_missing_fly_errors_with_docs_url_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # brew is useless advice off macOS; Linux gets the official install docs URL.
    monkeypatch.setattr("sys.platform", "linux")
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--fly", "--yes"])
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "https://fly.io/docs/flyctl/install/" in flat
    assert "brew install flyctl" not in flat


def test_deploy_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=("vercel",), returncode=2)
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 2


def test_deploy_noninteractive_without_yes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, available=("vercel",), agentic=True)
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 2  # refusing to deploy without confirmation is a usage error
    assert "--yes" in result.output
    assert _cmds(calls) == []


@pytest.mark.parametrize("flag", ["--prod", "--vercel", "--railway", "--fly", "--yes"])
def test_deploy_help_lists_flags(flag: str) -> None:
    result = runner.invoke(app, ["deploy", "--help"])
    assert result.exit_code == 0
    assert flag in _ANSI.sub("", result.output)


def test_deploy_outside_project_errors_like_dev(
    in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same guard as `assembly dev`/`assembly share`: outside a scaffolded project, say
    # "run `assembly init`" — not "install the Vercel CLI".
    in_project.unlink()
    calls = _stub(monkeypatch, available=("vercel",))
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 1
    assert "No Procfile here (expected ./Procfile)" in result.output
    assert "assembly init" in result.output
    assert _cmds(calls) == []  # never deployed


def test_deploy_procfile_guard_runs_before_cli_check(
    in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # With neither a Procfile nor the Vercel CLI, the missing-project error must win:
    # the actionable next step is `assembly init`, not installing a deploy CLI.
    in_project.unlink()
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--yes"])
    assert result.exit_code == 1
    assert "No Procfile here" in result.output
    assert "Vercel CLI" not in result.output


def test_deploy_prod_usage_error_wins_even_outside_project(
    in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Flag validation precedes the project/CLI checks, with the conventional usage exit 2.
    in_project.unlink()
    _stub(monkeypatch, available=())
    result = runner.invoke(app, ["deploy", "--fly", "--prod", "--yes"])
    assert result.exit_code == 2
    assert "--prod is only supported for Vercel deploys." in result.output
    assert "No Procfile" not in result.output


def test_deploy_prod_error_suggests_dropping_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, available=("railway",))
    result = runner.invoke(app, ["deploy", "--railway", "--prod", "--yes"])
    flat = " ".join(result.output.split())
    assert "Drop --prod, or drop --railway to deploy to Vercel." in flat


def test_install_hint_platform_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from aai_cli.commands.deploy import _exec as deploy

    monkeypatch.setattr("sys.platform", "darwin")
    assert deploy._install_hint(FLY) == "Install it with `brew install flyctl`."
    monkeypatch.setattr("sys.platform", "linux")
    assert deploy._install_hint(FLY) == "Install it: https://fly.io/docs/flyctl/install/"
    # npm-based targets have one hint that works everywhere, on either platform.
    assert deploy._install_hint(VERCEL) == "Install it with `npm i -g vercel`."
    assert deploy._install_hint(RAILWAY) == "Install it with `npm i -g @railway/cli`."
