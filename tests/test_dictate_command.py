"""The `assembly dictate` Typer surface: argv -> DictateOptions mapping. Session
behavior lives in test_dictate_exec.py."""

from typer.testing import CliRunner

from aai_cli.commands.dictate import _exec as dictate_exec
from aai_cli.main import app

runner = CliRunner()


def _capture_run(monkeypatch):
    seen = {}

    def fake_run(opts, state, *, json_mode):
        seen["opts"] = opts
        seen["json_mode"] = json_mode

    monkeypatch.setattr(dictate_exec, "run_dictate", fake_run)
    return seen


def test_defaults_map_to_options(monkeypatch):
    seen = _capture_run(monkeypatch)
    result = runner.invoke(app, ["dictate"])
    assert result.exit_code == 0
    assert seen["opts"] == dictate_exec.DictateOptions(
        language=None,
        prompt=None,
        word_boost=None,
        device=None,
        once=False,
        max_seconds=120.0,
    )
    assert seen["json_mode"] is False


def test_every_flag_maps_to_its_option_field(monkeypatch):
    seen = _capture_run(monkeypatch)
    result = runner.invoke(
        app,
        [
            "dictate",
            "--language",
            "es",
            "--prompt",
            "Verbatim.",
            "--word-boost",
            "AssemblyAI",
            "--word-boost",
            "LeMUR",
            "--device",
            "2",
            "--once",
            "--max-seconds",
            "30",
            "--json",
        ],
    )
    assert result.exit_code == 0
    assert seen["opts"] == dictate_exec.DictateOptions(
        language="es",
        prompt="Verbatim.",
        word_boost=["AssemblyAI", "LeMUR"],
        device=2,
        once=True,
        max_seconds=30.0,
    )
    assert seen["json_mode"] is True


def test_max_seconds_is_capped_at_the_api_limit():
    result = runner.invoke(app, ["dictate", "--max-seconds", "200"])
    assert result.exit_code == 2
    assert "120" in result.output
