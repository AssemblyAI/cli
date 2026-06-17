from __future__ import annotations

import dataclasses
import json as json_mod
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.app import coding_agent
from aai_cli.commands.code import _exec as code_exec
from aai_cli.core import environments
from aai_cli.main import app
from tests._snapshot_surface import normalize

runner = CliRunner()

_GATEWAY = environments.get(environments.DEFAULT_ENV).llm_gateway_base


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    has_opencode: bool = True,
    returncode: int = 0,
    assemblyai_skill: str | None = None,
) -> dict[str, object]:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    # Isolate the skills root so the conventions reflect only what the test sets up,
    # never the host's ~/.claude (coding_agent.skills_root honors CLAUDE_CONFIG_DIR).
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    if assemblyai_skill is not None:
        skill_md = coding_agent.skill_dir() / "SKILL.md"
        skill_md.parent.mkdir(parents=True, exist_ok=True)
        skill_md.write_text(assemblyai_skill, encoding="utf-8")
    monkeypatch.setattr(
        "shutil.which",
        lambda name: (
            f"/usr/bin/{name}" if has_opencode and name == code_exec.OPENCODE_BIN else None
        ),
    )
    calls: dict[str, object] = {}

    def fake_run(cmd: list[str], *, env: dict[str, str], check: bool) -> types.SimpleNamespace:
        calls["cmd"] = cmd
        calls["env"] = env
        calls["check"] = check
        # Read the generated config + instructions while the temp dir still exists.
        cfg_path = env.get("OPENCODE_CONFIG")
        if cfg_path:
            cfg = json_mod.loads(Path(cfg_path).read_text(encoding="utf-8"))
            calls["config"] = cfg
            instructions = cfg.get("instructions") or []
            if instructions:
                calls["instructions_text"] = Path(instructions[0]).read_text(encoding="utf-8")
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.code._exec.subprocess.run", fake_run)
    return calls


def _config(calls: dict[str, object]) -> dict[str, object]:
    cfg = calls["config"]
    assert isinstance(cfg, dict)
    return cfg


def _dict(value: object) -> dict[str, object]:
    """Assert ``value`` is a dict and return it (narrows the object-typed parsed JSON)."""
    assert isinstance(value, dict)
    return value


def _list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def test_code_options_are_frozen() -> None:
    # CodeOptions is parsed argv handed to run_code; freezing guards against a body
    # mutating the request it was given.
    opts = code_exec.CodeOptions(model="m", message=None)
    field = "model"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(opts, field, "tampered")


def test_code_launches_opencode_interactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    # No --message: launch the interactive TUI (no `run` subcommand).
    assert calls["cmd"] == ["opencode"]
    # check=False: opencode's own non-zero exit is surfaced by us, not raised by subprocess.
    assert calls["check"] is False


def test_code_config_wires_gateway_provider_and_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    cfg = _config(calls)
    provider = _dict(_dict(cfg["provider"])["assemblyai"])
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    options = _dict(provider["options"])
    assert options["baseURL"] == _GATEWAY
    # The key rides the env via {env:…}, never written into the config file.
    assert options["apiKey"] == "{env:ASSEMBLYAI_API_KEY}"
    assert provider["models"] == {"claude-opus-4-7": {"name": "claude-opus-4-7"}}
    assert cfg["model"] == "assemblyai/claude-opus-4-7"


def test_code_config_registers_docs_mcp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    mcp = _dict(_config(calls)["mcp"])["assemblyai-docs"]
    assert mcp == {"type": "remote", "url": "https://mcp.assemblyai.com/docs", "enabled": True}


def test_code_config_disables_autoupdate_and_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    cfg = _config(calls)
    assert cfg["autoupdate"] is False
    assert cfg["share"] == "disabled"


def test_code_passes_key_and_config_path_in_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    env = calls["env"]
    assert isinstance(env, dict)
    assert env["ASSEMBLYAI_API_KEY"] == "sk_test"
    # OPENCODE_CONFIG points opencode at our generated config.
    assert Path(env["OPENCODE_CONFIG"]).name == "opencode.json"


def test_code_custom_model_is_namespaced_to_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "--model", "gpt-5.1"])
    assert result.exit_code == 0, result.output
    cfg = _config(calls)
    assert cfg["model"] == "assemblyai/gpt-5.1"
    provider = _dict(_dict(cfg["provider"])["assemblyai"])
    assert provider["models"] == {"gpt-5.1": {"name": "gpt-5.1"}}


def test_code_injects_bundled_aai_cli_skill_as_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The aai-cli skill ships in the wheel, so it is always wired in, even with no skills
    # installed on disk.
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    assert len(_list(_config(calls)["instructions"])) == 1
    text = calls["instructions_text"]
    assert isinstance(text, str)
    assert "## aai-cli" in text
    assert "Use the AssemblyAI CLI" in text  # from the bundled SKILL.md
    assert "## assemblyai" not in text  # not installed in this test


def test_code_injects_installed_assemblyai_skill_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(
        monkeypatch, tmp_path, assemblyai_skill="---\nname: assemblyai\n---\nTRANSCRIBE-WITH-SDK\n"
    )
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    text = calls["instructions_text"]
    assert isinstance(text, str)
    assert "## aai-cli" in text
    assert "## assemblyai" in text
    assert "TRANSCRIBE-WITH-SDK" in text


def test_code_message_runs_one_shot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "-m", "add a test"])
    assert result.exit_code == 0, result.output
    # `opencode run <message>` runs the instruction non-interactively and exits.
    assert calls["cmd"] == ["opencode", "run", "add a test"]


def test_code_missing_opencode_errors_without_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path, has_opencode=False)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 1
    # Console soft-wraps the hint, so normalize whitespace before matching.
    flat = " ".join(normalize(result.output).split())
    assert "opencode is required" in flat
    assert "npm i -g opencode-ai" in flat
    assert "for install options" in flat  # the suggestion line
    assert "cmd" not in calls  # never reached subprocess.run


def test_code_nonzero_exit_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub(monkeypatch, tmp_path, returncode=2)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 2


def test_code_human_launch_note_is_plain_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    assert "Launching opencode via the AssemblyAI LLM Gateway (model claude-opus-4-7)" in (
        result.stdout
    )
    assert '"status"' not in result.stdout  # human mode prints text, not the JSON record


def test_code_json_launch_record_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "--json"])
    assert result.exit_code == 0, result.output
    assert json_mod.loads(result.stdout) == {
        "status": "launching",
        "tool": "opencode",
        "model": "claude-opus-4-7",
        "gateway": _GATEWAY,
    }


def test_bundled_cli_skill_doc_reads_packaged_skill() -> None:
    # Directly cover the wheel-resource read so a packaging regression fails loudly.
    doc = coding_agent.bundled_cli_skill_doc()
    assert "name: aai-cli" in doc
    assert "Use the AssemblyAI CLI" in doc
