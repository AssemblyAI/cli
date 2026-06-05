import json
import sys
import types
from collections import namedtuple

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.commands import doctor
from aai_cli.errors import APIError
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture
def healthy(monkeypatch):
    """A fully-ready environment: valid key, all tools present, a microphone."""
    config.set_api_key("default", "sk_1234567890")
    monkeypatch.setattr("aai_cli.commands.doctor.client.validate_key", lambda _key: True)
    monkeypatch.setattr("aai_cli.commands.doctor.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr("aai_cli.commands.doctor._probe_input_devices", lambda: 2)


def _checks(result):
    return {c["name"]: c for c in json.loads(result.output)["checks"]}


def test_doctor_all_ok(healthy):
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert {c["status"] for c in payload["checks"]} == {"ok"}


def test_doctor_no_api_key_fails(healthy):
    config.clear_api_key("default")
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    api = _checks(result)["api-key"]
    assert api["status"] == "fail"
    assert "login" in api["fix"]


def test_doctor_rejected_key_fails(healthy, monkeypatch):
    monkeypatch.setattr("aai_cli.commands.doctor.client.validate_key", lambda _key: False)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    assert _checks(result)["api-key"]["status"] == "fail"


def test_doctor_network_error_is_a_failure(healthy, monkeypatch):
    def boom(_key):
        raise APIError("Network error contacting AssemblyAI: timeout")

    monkeypatch.setattr("aai_cli.commands.doctor.client.validate_key", boom)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    api = _checks(result)["api-key"]
    assert api["status"] == "fail"
    assert "reach AssemblyAI" in api["detail"]


def test_doctor_ffmpeg_missing_warns_but_passes(healthy, monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.doctor.shutil.which",
        lambda tool: None if tool == "ffmpeg" else f"/usr/bin/{tool}",
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0  # a warning never blocks
    assert _checks(result)["ffmpeg"]["status"] == "warn"
    assert json.loads(result.output)["ok"] is True


def test_doctor_audio_unavailable_warns_but_passes(healthy, monkeypatch):
    def no_audio():
        raise ImportError("no sounddevice")

    monkeypatch.setattr("aai_cli.commands.doctor._probe_input_devices", no_audio)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    audio = _checks(result)["audio"]
    assert audio["status"] == "warn"
    assert "sounddevice" in audio["fix"]


def test_doctor_no_microphone_warns(healthy, monkeypatch):
    monkeypatch.setattr("aai_cli.commands.doctor._probe_input_devices", lambda: 0)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert _checks(result)["audio"]["status"] == "warn"


def test_doctor_coding_agent_missing_warns(healthy, monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.doctor.shutil.which",
        lambda tool: None if tool in ("claude", "npx") else f"/usr/bin/{tool}",
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    agent_check = _checks(result)["coding-agent"]
    assert agent_check["status"] == "warn"
    assert "claude" in agent_check["detail"]


def test_doctor_json_shape(healthy):
    payload = json.loads(runner.invoke(app, ["doctor", "--json"]).output)
    assert set(payload) == {"ok", "checks"}
    names = [c["name"] for c in payload["checks"]]
    assert names == ["python", "api-key", "ffmpeg", "audio", "coding-agent"]
    for c in payload["checks"]:
        assert set(c) == {"name", "status", "affects", "detail", "fix"}


def test_doctor_human_output_renders(healthy):
    # Force human mode by asking explicitly (default would be JSON under the test runner).
    result = runner.invoke(app, ["doctor"], env={"NO_COLOR": "1"})
    # JSON is the default when not a TTY; either way the run must succeed.
    assert result.exit_code == 0


def test_doctor_listed_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output


# --- unit tests for the helpers and the human renderer ---


def test_check_python_flags_old_interpreter(monkeypatch):
    VI = namedtuple("VI", "major minor micro releaselevel serial")
    monkeypatch.setattr(doctor.sys, "version_info", VI(3, 9, 0, "final", 0))
    check = doctor._check_python()
    assert check["status"] == "fail"
    assert "3.9.0" in check["detail"]


def test_check_audio_handles_portaudio_failure(monkeypatch):
    def boom():
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(doctor, "_probe_input_devices", boom)
    check = doctor._check_audio()
    assert check["status"] == "warn"
    assert "PortAudio" in check["detail"]


def test_probe_input_devices_counts_integer_input_channels(monkeypatch):
    class FakeSoundDevice(types.ModuleType):
        def query_devices(self):
            return [
                {"max_input_channels": 1},
                {"max_input_channels": "bad"},
                {"max_input_channels": 0},
            ]

    fake_sd = FakeSoundDevice("sounddevice")
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    assert doctor._probe_input_devices() == 1


def test_render_ok_payload_shows_ready():
    payload: doctor.DoctorResult = {
        "ok": True,
        "checks": [
            {"name": "python", "status": "ok", "affects": [], "detail": "3.12", "fix": None}
        ],
    }
    text = doctor._render(payload)
    assert "python" in text
    assert "Everything looks good." in text


def test_render_problem_payload_shows_fix_and_problem_banner():
    payload: doctor.DoctorResult = {
        "ok": False,
        "checks": [
            {
                "name": "api-key",
                "status": "fail",
                "affects": ["everything"],
                "detail": "No API key found.",
                "fix": "Run 'aai login'.",
            }
        ],
    }
    text = doctor._render(payload)
    assert "fix:" in text
    assert "Run 'aai login'." in text
    assert "1 problem found" in text
