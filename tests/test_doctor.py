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
    # The MCP probe shells out to `claude mcp get`; keep the suite hermetic and
    # report the full setup (docs MCP + both skills) as installed.
    monkeypatch.setattr("aai_cli.commands.doctor.coding_agent.missing_components", list)


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


def test_doctor_no_keyring_recommends_env_var(healthy, monkeypatch):
    # On a box with no usable keyring, `assembly login` can't persist a key either, so the
    # fix must point at ASSEMBLYAI_API_KEY rather than a dead-end browser login.
    config.clear_api_key("default")
    monkeypatch.setattr("aai_cli.commands.doctor.config.keyring_usable", lambda: False)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    api = _checks(result)["api-key"]
    assert api["status"] == "fail"
    assert "ASSEMBLYAI_API_KEY" in api["fix"]
    assert "no usable OS keyring" in api["detail"]


def test_doctor_success_suggests_trying_transcribe(healthy, monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "assembly transcribe --sample" in result.output


def test_doctor_rejected_key_fails(healthy, monkeypatch):
    monkeypatch.setattr("aai_cli.commands.doctor.client.validate_key", lambda _key: False)
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    api = _checks(result)["api-key"]
    assert api["status"] == "fail"
    # validate_key collapses every auth-shaped failure (401, 403, proxy "forbidden")
    # to False, so the detail must not claim a status code that was never observed.
    assert api["detail"] == "API key was rejected by the server."
    assert "401" not in api["detail"]
    assert "assembly login" in api["fix"]


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


def test_doctor_coding_agent_fully_set_up_does_not_suggest_install(healthy):
    # Regression: with the docs MCP + skills already installed, doctor must say so
    # instead of telling the user to run a setup that's already done.
    result = runner.invoke(app, ["doctor", "--json"])
    agent_check = _checks(result)["coding-agent"]
    assert agent_check["status"] == "ok"
    assert agent_check["detail"] == "claude and npx found; docs MCP + skills installed."


def test_doctor_coding_agent_not_set_up_names_whats_missing(healthy, monkeypatch):
    monkeypatch.setattr(
        "aai_cli.commands.doctor.coding_agent.missing_components",
        lambda: ["docs MCP", "aai-cli skill"],
    )
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0  # an un-run setup never blocks
    agent_check = _checks(result)["coding-agent"]
    assert agent_check["status"] == "ok"
    assert agent_check["detail"] == (
        "claude and npx found; run 'assembly setup install' to add: docs MCP, aai-cli skill."
    )


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
    assert set(payload) == {"ok", "profile", "environment", "checks"}
    names = [c["name"] for c in payload["checks"]]
    assert names == ["python", "api-key", "ffmpeg", "audio", "coding-agent"]
    for c in payload["checks"]:
        assert set(c) == {"name", "status", "affects", "detail", "fix"}


def test_doctor_json_reports_profile_and_environment(healthy):
    payload = json.loads(runner.invoke(app, ["doctor", "--json"]).output)
    assert payload["profile"] == "default"
    assert payload["environment"] == "production"


def test_doctor_json_reports_selected_env_and_profile(healthy):
    payload = json.loads(
        runner.invoke(app, ["--env", "sandbox000", "-p", "default", "doctor", "--json"]).output
    )
    assert payload["environment"] == "sandbox000"
    assert payload["profile"] == "default"


def test_doctor_network_fix_names_active_env_host(healthy, monkeypatch):
    # Under --sandbox the fix must point at the sandbox API host, not hardcode
    # api.assemblyai.com (which being reachable wouldn't help a sandbox user).
    def boom(_key):
        raise APIError("Network error contacting AssemblyAI: timeout")

    monkeypatch.setattr("aai_cli.commands.doctor.client.validate_key", boom)
    result = runner.invoke(app, ["--env", "sandbox000", "doctor", "--json"])
    fix = _checks(result)["api-key"]["fix"]
    assert "that api.sandbox000.assemblyai-labs.com is reachable" in fix
    assert "api.assemblyai.com" not in fix
    assert "https://" not in fix  # the scheme is stripped: it's a host, not a URL

    prod = runner.invoke(app, ["doctor", "--json"])
    assert "that api.assemblyai.com is reachable" in _checks(prod)["api-key"]["fix"]


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
    check = doctor.check_python()
    assert check["status"] == "fail"
    assert "3.9.0" in check["detail"]
    assert check["affects"] == ["everything"]


def test_check_audio_handles_portaudio_failure(monkeypatch):
    def boom():
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(doctor, "_probe_input_devices", boom)
    check = doctor.check_audio()
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


def test_render_ok_payload_shows_ready() -> None:
    payload: doctor.DoctorResult = {
        "ok": True,
        "profile": "default",
        "environment": "production",
        "checks": [
            {"name": "python", "status": "ok", "affects": [], "detail": "3.12", "fix": None}
        ],
    }
    text = doctor.render(payload)
    assert "python" in text
    assert "Everything looks good." in text
    assert "assembly transcribe --sample" in text  # the next-step hint (profile present)


def test_render_reports_profile_and_environment_line() -> None:
    payload: doctor.DoctorResult = {
        "ok": True,
        "profile": "staging",
        "environment": "sandbox000",
        "checks": [
            {"name": "python", "status": "ok", "affects": [], "detail": "3.12", "fix": None}
        ],
    }
    text = doctor.render(payload)
    assert "profile: staging" in text
    assert "environment: sandbox000" in text


def test_render_omits_profile_line_for_partial_payloads() -> None:
    # The onboarding wizard reuses render for a quick environment check with no
    # profile/environment context — no half-empty "profile:" line may appear.
    payload: doctor.DoctorResult = {
        "ok": True,
        "checks": [
            {"name": "python", "status": "ok", "affects": [], "detail": "3.12", "fix": None}
        ],
    }
    text = doctor.render(payload)
    assert "profile:" not in text
    assert "environment:" not in text
    # The wizard reuses render() and has its own next-steps, so the "try transcribe"
    # hint must NOT appear on a profile-less partial payload.
    assert "assembly transcribe --sample" not in text


def test_doctor_human_output_shows_profile_and_environment(healthy, monkeypatch):
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "profile: default" in result.output
    assert "environment: production" in result.output


def test_render_problem_payload_shows_fix_and_problem_banner() -> None:
    payload: doctor.DoctorResult = {
        "ok": False,
        "profile": "default",
        "environment": "production",
        "checks": [
            {
                "name": "api-key",
                "status": "fail",
                "affects": ["everything"],
                "detail": "No API key found.",
                "fix": "Run 'assembly login'.",
            }
        ],
    }
    text = doctor.render(payload)
    assert "fix:" in text
    assert "Run 'assembly login'." in text
    assert "1 problem found" in text
