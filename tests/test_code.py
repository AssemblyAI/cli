from __future__ import annotations

import dataclasses
import json as json_mod
import types

import pytest
from typer.testing import CliRunner

from aai_cli.commands.code import _exec as code_exec
from aai_cli.core import environments
from aai_cli.main import app
from tests._snapshot_surface import normalize

runner = CliRunner()

_GATEWAY = environments.get(environments.DEFAULT_ENV).llm_gateway_base


def test_code_options_are_frozen() -> None:
    # CodeOptions is parsed argv handed to run_code; freezing guards against a body
    # mutating the request it was given.
    opts = code_exec.CodeOptions(model="m", files=())
    field = "model"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(opts, field, "tampered")


def _stub(
    monkeypatch: pytest.MonkeyPatch, *, has_aider: bool = True, returncode: int = 0
) -> dict[str, object]:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: f"/usr/bin/{name}" if has_aider and name == code_exec.AIDER_BIN else None,
    )
    calls: dict[str, object] = {}

    def fake_run(cmd: list[str], *, env: dict[str, str], check: bool) -> types.SimpleNamespace:
        calls["cmd"] = cmd
        calls["env"] = env
        calls["check"] = check
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.code._exec.subprocess.run", fake_run)
    return calls


def test_code_launches_aider_with_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["aider", "--model", "openai/claude-opus-4-7"]
    # check=False: aider's own non-zero exit is surfaced by us, not raised by subprocess.
    assert calls["check"] is False


def test_code_wires_gateway_and_key_into_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    env = calls["env"]
    assert isinstance(env, dict)
    # aider/litellm read these to reach an OpenAI-compatible endpoint.
    assert env["OPENAI_API_BASE"] == _GATEWAY
    assert env["OPENAI_API_KEY"] == "sk_test"


def test_code_custom_model_is_prefixed_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch)
    result = runner.invoke(app, ["code", "--model", "gpt-5.1"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == ["aider", "--model", "openai/gpt-5.1"]


def test_code_passes_files_positionally(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch)
    result = runner.invoke(app, ["code", "api/index.py", "README.md"])
    assert result.exit_code == 0, result.output
    assert calls["cmd"] == [
        "aider",
        "--model",
        "openai/claude-opus-4-7",
        "api/index.py",
        "README.md",
    ]


def test_code_missing_aider_errors_without_launching(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub(monkeypatch, has_aider=False)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 1
    # Console soft-wraps the hint, so normalize whitespace before matching.
    flat = " ".join(normalize(result.output).split())
    assert "aider is required" in flat
    assert "uv tool install aider-chat" in flat
    assert "for install options" in flat  # the suggestion line
    assert "cmd" not in calls  # never reached subprocess.run


def test_code_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, returncode=2)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 2


def test_code_human_launch_note_is_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    assert "Launching aider via the AssemblyAI LLM Gateway (model claude-opus-4-7)" in (
        result.stdout
    )
    assert '"status"' not in result.stdout  # human mode prints text, not the JSON record


def test_code_json_launch_record_is_machine_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch)
    result = runner.invoke(app, ["code", "--json"])
    assert result.exit_code == 0, result.output
    assert json_mod.loads(result.stdout) == {
        "status": "launching",
        "tool": "aider",
        "model": "claude-opus-4-7",
        "gateway": _GATEWAY,
    }
