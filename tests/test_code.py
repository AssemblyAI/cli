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
from aai_cli.ui import theme
from tests._snapshot_surface import normalize

runner = CliRunner()

_GATEWAY = environments.get(environments.DEFAULT_ENV).llm_gateway_base


def _stub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    has_aider: bool = True,
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
        lambda name: f"/usr/bin/{name}" if has_aider and name == code_exec.AIDER_BIN else None,
    )
    calls: dict[str, object] = {}

    def fake_run(cmd: list[str], *, env: dict[str, str], check: bool) -> types.SimpleNamespace:
        calls["cmd"] = cmd
        calls["env"] = env
        calls["check"] = check
        if "--read" in cmd:
            # Read the conventions file while the temp dir still exists (during the call).
            calls["conventions"] = Path(cmd[cmd.index("--read") + 1]).read_text(encoding="utf-8")
        return types.SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("aai_cli.commands.code._exec.subprocess.run", fake_run)
    return calls


def _base_argv(cmd: object) -> list[str]:
    """The launch argv up to (excluding) the ``--read <conventions>`` pair and beyond."""
    assert isinstance(cmd, list)
    return cmd[: cmd.index("--read")] if "--read" in cmd else cmd


def _prefix(model: str) -> list[str]:
    """The fixed flag prefix every launch carries: models, quiet warnings, brand theme."""
    return [
        "aider",
        "--model",
        f"openai/{model}",
        "--weak-model",
        f"openai/{code_exec.WEAK_MODEL}",
        "--no-show-model-warnings",
        # Brand theme mapped from ui/theme.py.
        "--user-input-color",
        theme.BRAND,
        "--assistant-output-color",
        theme.ACCENT,
        "--tool-output-color",
        theme.MUTED,
        "--tool-error-color",
        theme.ERROR,
        "--code-theme",
        "ansi_dark",
    ]


def test_code_options_are_frozen() -> None:
    # CodeOptions is parsed argv handed to run_code; freezing guards against a body
    # mutating the request it was given.
    opts = code_exec.CodeOptions(model="m", files=(), message=None)
    field = "model"
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(opts, field, "tampered")


def test_code_launches_aider_with_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    # Default launch: main model + cheap weak model (the gateway's default) + quiet warnings.
    assert _base_argv(calls["cmd"]) == _prefix("claude-opus-4-7")
    assert code_exec.WEAK_MODEL != "claude-opus-4-7"  # weak model is the cheaper one
    # check=False: aider's own non-zero exit is surfaced by us, not raised by subprocess.
    assert calls["check"] is False


def test_code_wires_gateway_and_key_into_child_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    env = calls["env"]
    assert isinstance(env, dict)
    # aider/litellm read these to reach an OpenAI-compatible endpoint.
    assert env["OPENAI_API_BASE"] == _GATEWAY
    assert env["OPENAI_API_KEY"] == "sk_test"
    # aider's own analytics + update notifier are silenced (the CLI owns both).
    assert env["AIDER_ANALYTICS"] == "false"
    assert env["AIDER_CHECK_UPDATE"] == "false"


def test_code_custom_model_is_prefixed_openai(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "--model", "gpt-5.1"])
    assert result.exit_code == 0, result.output
    assert _base_argv(calls["cmd"]) == _prefix("gpt-5.1")


def test_code_passes_files_positionally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "api/index.py", "README.md"])
    assert result.exit_code == 0, result.output
    # Files come after the flag prefix and before the --read conventions pair.
    assert _base_argv(calls["cmd"]) == [*_prefix("claude-opus-4-7"), "api/index.py", "README.md"]


def test_code_message_runs_one_shot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code", "-m", "add a test"])
    assert result.exit_code == 0, result.output
    cmd = calls["cmd"]
    assert isinstance(cmd, list)
    # --message is appended last so aider runs the instruction non-interactively and exits.
    assert cmd[-2:] == ["--message", "add a test"]


def test_code_no_message_omits_the_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    cmd = calls["cmd"]
    assert isinstance(cmd, list)
    assert "--message" not in cmd


def test_code_injects_bundled_aai_cli_skill_as_conventions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The aai-cli skill ships in the wheel, so it is always read into aider as context,
    # even with no skills installed on disk.
    calls = _stub(monkeypatch, tmp_path)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    cmd = calls["cmd"]
    assert isinstance(cmd, list)
    assert "--read" in cmd
    conventions = calls["conventions"]
    assert isinstance(conventions, str)
    assert "## aai-cli" in conventions
    assert "Use the AssemblyAI CLI" in conventions  # from the bundled SKILL.md
    assert "## assemblyai" not in conventions  # not installed in this test


def test_code_injects_installed_assemblyai_skill_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(
        monkeypatch, tmp_path, assemblyai_skill="---\nname: assemblyai\n---\nTRANSCRIBE-WITH-SDK\n"
    )
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 0, result.output
    conventions = calls["conventions"]
    assert isinstance(conventions, str)
    # Both skills land in the one conventions file when the assemblyai skill is present.
    assert "## aai-cli" in conventions
    assert "## assemblyai" in conventions
    assert "TRANSCRIBE-WITH-SDK" in conventions


def test_code_missing_aider_errors_without_launching(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub(monkeypatch, tmp_path, has_aider=False)
    result = runner.invoke(app, ["code"])
    assert result.exit_code == 1
    # Console soft-wraps the hint, so normalize whitespace before matching.
    flat = " ".join(normalize(result.output).split())
    assert "aider is required" in flat
    assert "uv tool install aider-chat" in flat
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
    assert "Launching aider via the AssemblyAI LLM Gateway (model claude-opus-4-7)" in (
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
        "tool": "aider",
        "model": "claude-opus-4-7",
        "gateway": _GATEWAY,
    }


def test_bundled_cli_skill_doc_reads_packaged_skill() -> None:
    # Directly cover the wheel-resource read so a packaging regression fails loudly.
    doc = coding_agent.bundled_cli_skill_doc()
    assert "name: aai-cli" in doc
    assert "Use the AssemblyAI CLI" in doc
