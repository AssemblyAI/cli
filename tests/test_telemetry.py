"""Unit tests for the telemetry core (aai_cli/telemetry.py) and its config state."""

import json
import subprocess
import sys
import time
import uuid

import pytest
import typer

from aai_cli import config, telemetry
from aai_cli.errors import CLIError, UsageError

# --- token / url resolution -------------------------------------------------


def test_shipped_token_is_a_write_only_client_token(neutralize_shipped_token):
    # The committed credential must be a Datadog *client* token (pub…, write-only,
    # embeddable by design) — never an API key. The autouse fixture blanks it for
    # the suite and hands back the real value for exactly this assertion.
    assert neutralize_shipped_token == "pub0d633113b9f7d22faff215fefaf30b43"
    # Blanked in-suite, so nothing else accidentally goes live:
    assert telemetry.client_token() == ""


def test_client_token_env_overrides_shipped(monkeypatch):
    monkeypatch.setattr(telemetry, "SHIPPED_CLIENT_TOKEN", "pub_shipped")
    assert telemetry.client_token() == "pub_shipped"
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_env")
    assert telemetry.client_token() == "pub_env"


def test_intake_url_default_and_override(monkeypatch):
    assert telemetry.intake_url() == "https://browser-intake-datadoghq.com/api/v2/logs"
    monkeypatch.setenv(telemetry.ENV_INTAKE_URL, "https://example.test/logs")
    assert telemetry.intake_url() == "https://example.test/logs"


# --- consent ------------------------------------------------------------------


def test_consent_granted_by_default():
    assert telemetry.consent_granted() is True


@pytest.mark.parametrize("var", ["AAI_TELEMETRY_DISABLED", "DO_NOT_TRACK"])
@pytest.mark.parametrize("value", ["1", "true"])
def test_consent_env_kill_switches(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    assert telemetry.consent_granted() is False


def test_consent_follows_persisted_choice():
    config.set_telemetry_enabled(enabled=False)
    assert telemetry.consent_granted() is False
    config.set_telemetry_enabled(enabled=True)
    assert telemetry.consent_granted() is True


def test_consent_env_wins_over_persisted_enable(monkeypatch):
    config.set_telemetry_enabled(enabled=True)
    monkeypatch.setenv(telemetry.ENV_DISABLED, "1")
    assert telemetry.consent_granted() is False


def test_is_enabled_requires_token_and_consent(monkeypatch):
    assert telemetry.is_enabled() is False  # consent alone is not enough
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_test")
    assert telemetry.is_enabled() is True
    config.set_telemetry_enabled(enabled=False)
    assert telemetry.is_enabled() is False


# --- config-backed telemetry state -------------------------------------------


def test_telemetry_enabled_roundtrip():
    assert config.get_telemetry_enabled() is None
    config.set_telemetry_enabled(enabled=False)
    assert config.get_telemetry_enabled() is False
    config.set_telemetry_enabled(enabled=True)
    assert config.get_telemetry_enabled() is True


def test_device_id_is_a_stable_random_uuid(tmp_config):
    first = config.get_device_id()
    # A parseable UUID, not something derived from the machine or account.
    assert str(uuid.UUID(first)) == first
    assert config.get_device_id() == first
    # Persisted: a fresh load (new process semantics) sees the same id.
    assert f'device_id = "{first}"' in (tmp_config / "config.toml").read_text()


def test_device_id_does_not_clobber_other_settings():
    config.set_profile_env("default", "production")
    config.get_device_id()
    assert config.get_profile_env("default") == "production"


# --- event shape ----------------------------------------------------------------


def test_build_event_is_allowlisted_and_exact(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.9")
    event = telemetry.build_event("aai transcribe", outcome="success", exit_code=0, duration_ms=250)
    from aai_cli import __version__

    assert event == {
        "ddsource": "aai-cli",
        "service": "aai-cli",
        "ddtags": f"version:{__version__}",
        "message": "aai transcribe success",
        "command": "aai transcribe",
        "outcome": "success",
        "exit_code": 0,
        "duration_ms": 250,
        "cli_version": __version__,
        "os": "linux",
        "python_version": "3.12.9",
        "ci": False,
        "device_id": config.get_device_id(),
    }


def test_build_event_marks_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    event = telemetry.build_event("aai stream", outcome="api_error", exit_code=1, duration_ms=5)
    assert event["ci"] is True
    assert event["outcome"] == "api_error"
    assert event["exit_code"] == 1


# --- dispatch (detached flusher handoff) ------------------------------------


def test_dispatch_spawns_detached_flusher(monkeypatch):
    calls = {}

    def fake_popen(argv, **kwargs):
        calls["argv"], calls["kwargs"] = argv, kwargs

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_test")
    telemetry.dispatch({"command": "aai doctor"})

    # The child is the CLI's own (hidden) `telemetry flush` subcommand — explicit
    # plumbing, not an inline -c snippet.
    assert calls["argv"][:5] == [sys.executable, "-m", "aai_cli", "telemetry", "flush"]
    payload = json.loads(calls["argv"][5])
    assert payload == {
        "url": "https://browser-intake-datadoghq.com/api/v2/logs",
        "token": "pub_test",
        "event": {"command": "aai doctor"},
    }
    # Detached with stdio discarded: the flusher can never block or pollute the
    # command. Telemetry is disabled in the child so a flush never spawns a flusher.
    kwargs = calls["kwargs"]
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert kwargs["env"]["AAI_TELEMETRY_DISABLED"] == "1"
    assert sorted(kwargs) == ["env", "start_new_session", "stderr", "stdout"]


# --- flusher ------------------------------------------------------------------


class _FakeClient:
    def __init__(self, record):
        self._record = record

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self._record["url"] = url
        self._record["kwargs"] = kwargs


def test_flush_payload_posts_to_intake(monkeypatch):
    record = {}

    def fake_client_factory(*, timeout):
        record["timeout"] = timeout
        return _FakeClient(record)

    monkeypatch.setattr("httpx2.Client", fake_client_factory)
    raw = json.dumps(
        {"url": "https://example.test/logs", "token": "pub_x", "event": {"command": "aai llm"}}
    )
    telemetry.flush_payload(raw)

    assert record["timeout"] == 5.0
    assert record["url"] == "https://example.test/logs"
    assert record["kwargs"] == {
        "params": {"dd-api-key": "pub_x"},
        "headers": {"DD-API-KEY": "pub_x"},
        "json": [{"command": "aai llm"}],
    }


def test_flush_command_delivers_payload(monkeypatch):
    # The hidden `aai telemetry flush` subcommand is what dispatch() spawns; drive it
    # through the real CLI so the spawned argv is known to be invocable end to end.
    from typer.testing import CliRunner

    from aai_cli.main import app

    seen = []
    monkeypatch.setattr(telemetry, "flush_payload", seen.append)
    result = CliRunner().invoke(app, ["telemetry", "flush", '{"the": "payload"}'])
    assert result.exit_code == 0
    assert seen == ['{"the": "payload"}']


# --- track --------------------------------------------------------------------


@pytest.fixture
def events(monkeypatch):
    """Enable telemetry and capture dispatched events instead of spawning flushers."""
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_test")
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    return captured


def _freeze_duration(monkeypatch, seconds):
    ticks = iter([100.0, 100.0 + seconds])
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))


def test_track_disabled_dispatches_nothing(monkeypatch):
    captured = []
    monkeypatch.setattr(telemetry, "dispatch", captured.append)
    ran = []
    with telemetry.track("aai doctor"):
        ran.append(True)
    assert ran == [True]
    assert captured == []


def test_track_success(events, monkeypatch):
    # 2.0s, not a sub-second value: 2.0 * 1000 = 2000 also catches an off-by-one
    # mutation of the ms factor, which int(0.25 * 1001) would round away.
    _freeze_duration(monkeypatch, 2.0)
    with telemetry.track("aai doctor"):
        pass
    (event,) = events
    assert event["command"] == "aai doctor"
    assert event["outcome"] == "success"
    assert event["exit_code"] == 0
    assert event["duration_ms"] == 2000


def test_track_cli_error_keeps_error_type_and_reraises(events):
    with pytest.raises(UsageError), telemetry.track("aai transcribe"):
        raise UsageError("bad flag")
    (event,) = events
    assert event["outcome"] == "usage_error"
    assert event["exit_code"] == 2


@pytest.mark.parametrize(
    ("code", "outcome"), [(0, "success"), (3, "error")], ids=["exit-0", "exit-3"]
)
def test_track_typer_exit_maps_code(events, code, outcome):
    with pytest.raises(typer.Exit), telemetry.track("aai login"):
        raise typer.Exit(code=code)
    (event,) = events
    assert event["outcome"] == outcome
    assert event["exit_code"] == code


def test_track_unexpected_exception_is_internal_error(events):
    with pytest.raises(RuntimeError), telemetry.track("aai stream"):
        raise RuntimeError("boom")
    (event,) = events
    assert event["outcome"] == "internal_error"
    assert event["exit_code"] == 1


@pytest.mark.parametrize("exc", [OSError("spawn failed"), CLIError("corrupt config")])
def test_track_send_failures_never_break_the_command(monkeypatch, exc):
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_test")

    def explode(event):
        raise exc

    monkeypatch.setattr(telemetry, "dispatch", explode)
    with telemetry.track("aai doctor"):
        pass  # must not raise
