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


# --- consent source -------------------------------------------------------------


def test_consent_source_default():
    assert telemetry.consent_source() == "default"


@pytest.mark.parametrize("enabled", [True, False])
def test_consent_source_persisted_choice(enabled):
    config.set_telemetry_enabled(enabled=enabled)
    assert telemetry.consent_source() == "config"


@pytest.mark.parametrize("var", ["AAI_TELEMETRY_DISABLED", "DO_NOT_TRACK"])
def test_consent_source_env_kill_switch_wins_over_config(monkeypatch, var):
    config.set_telemetry_enabled(enabled=True)  # the env switch outranks the choice
    monkeypatch.setenv(var, "1")
    assert telemetry.consent_source() == f"env:{var}"


def test_consent_source_env_ordering_matches_consent_granted(monkeypatch):
    monkeypatch.setenv("AAI_TELEMETRY_DISABLED", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert telemetry.consent_source() == "env:AAI_TELEMETRY_DISABLED"


# --- config-backed telemetry state -------------------------------------------


def test_has_device_id_probes_without_minting(tmp_config):
    assert config.has_device_id() is False
    # Probing must not itself mint/persist an id.
    config_file = tmp_config / "config.toml"
    assert not config_file.exists() or "device_id" not in config_file.read_text()
    config.get_device_id()
    assert config.has_device_id() is True


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
        "status": "info",
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


def test_build_event_success_is_info_with_no_error_attribute():
    # A success stays an info log and carries no `error` namespace, so it never
    # lands in Datadog Error Tracking.
    event = telemetry.build_event("aai transcribe", outcome="success", exit_code=0, duration_ms=1)
    assert event["status"] == "info"
    assert "error" not in event


def test_build_event_failure_feeds_error_tracking(monkeypatch):
    monkeypatch.setenv("CI", "true")
    event = telemetry.build_event("aai stream", outcome="api_error", exit_code=1, duration_ms=5)
    assert event["ci"] is True
    assert event["outcome"] == "api_error"
    assert event["exit_code"] == 1
    # status:error + the reserved error.kind are what promote it into Error Tracking;
    # error.kind mirrors the anonymous outcome. No message was provided, so none rides along.
    assert event["status"] == "error"
    assert event["error"] == {"kind": "api_error"}


def test_build_event_failure_carries_error_message():
    event = telemetry.build_event(
        "aai transcribe",
        outcome="api_error",
        exit_code=1,
        duration_ms=5,
        error_message="Audio file not found: clip.wav",
    )
    # error.message is the reserved attribute Error Tracking groups/displays on.
    assert event["error"] == {"kind": "api_error", "message": "Audio file not found: clip.wav"}


def test_build_event_error_message_capped_at_500_chars():
    # Exactly at the cap: untouched.
    exact = "y" * 500
    event = telemetry.build_event(
        "aai stream", outcome="api_error", exit_code=1, duration_ms=5, error_message=exact
    )
    assert event["error"] == {"kind": "api_error", "message": exact}
    # One over: truncated to exactly the cap.
    event = telemetry.build_event(
        "aai stream", outcome="api_error", exit_code=1, duration_ms=5, error_message="x" * 501
    )
    assert event["error"] == {"kind": "api_error", "message": "x" * 500}


def test_build_event_blank_error_message_is_omitted():
    # str(exc) can be "" (e.g. RuntimeError()); don't ship an empty message field.
    event = telemetry.build_event(
        "aai stream", outcome="internal_error", exit_code=1, duration_ms=5, error_message=""
    )
    assert event["error"] == {"kind": "internal_error"}


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
    # The hidden `assembly telemetry flush` subcommand is what dispatch() spawns; drive it
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


def _raise(exc: BaseException) -> None:
    # Raising via a call (not a literal `raise` in the `with` body) keeps the
    # assertions below visibly reachable to static analysis (CodeQL).
    raise exc


def test_track_cli_error_keeps_error_type_and_reraises(events):
    with pytest.raises(UsageError), telemetry.track("aai transcribe"):
        _raise(UsageError("bad flag"))
    (event,) = events
    assert event["outcome"] == "usage_error"
    assert event["exit_code"] == 2
    # The clean CLIError message the user saw rides along for Error Tracking.
    assert event["error"] == {"kind": "usage_error", "message": "bad flag"}


@pytest.mark.parametrize(
    ("code", "outcome"), [(0, "success"), (3, "error")], ids=["exit-0", "exit-3"]
)
def test_track_typer_exit_maps_code(events, code, outcome):
    with pytest.raises(typer.Exit), telemetry.track("aai login"):
        _raise(typer.Exit(code=code))
    (event,) = events
    assert event["outcome"] == outcome
    assert event["exit_code"] == code
    # A bare typer.Exit carries no message, so the failure event has only the kind.
    assert event.get("error") == ({"kind": "error"} if code else None)


def test_track_unexpected_exception_is_internal_error(events):
    with pytest.raises(RuntimeError), telemetry.track("aai stream"):
        _raise(RuntimeError("boom"))
    (event,) = events
    assert event["outcome"] == "internal_error"
    assert event["exit_code"] == 1
    assert event["error"] == {"kind": "internal_error", "message": "boom"}


@pytest.mark.parametrize("exc", [OSError("spawn failed"), CLIError("corrupt config")])
def test_track_send_failures_never_break_the_command(monkeypatch, exc):
    monkeypatch.setenv(telemetry.ENV_CLIENT_TOKEN, "pub_test")

    def explode(event):
        raise exc

    monkeypatch.setattr(telemetry, "dispatch", explode)
    with telemetry.track("aai doctor"):
        pass  # must not raise


# --- first-run disclosure -------------------------------------------------------


def test_first_run_notice_prints_once_on_device_id_mint(events, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["assembly", "doctor"])
    with telemetry.track("assembly doctor"):
        pass
    err = capsys.readouterr().err
    assert "Anonymous usage data is collected" in err
    assert "'assembly telemetry disable'" in err
    assert "DO_NOT_TRACK=1" in err
    assert err.count("Anonymous usage data") == 1
    # The device id persists, so a second run stays silent — at most once ever.
    with telemetry.track("assembly doctor"):
        pass
    assert "Anonymous usage data" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [["assembly", "-q", "doctor"], ["assembly", "transcribe", "x.wav", "--json"]],
    ids=["quiet", "json"],
)
def test_first_run_notice_suppressed_for_quiet_and_json(events, monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with telemetry.track("assembly doctor"):
        pass
    assert "Anonymous usage data" not in capsys.readouterr().err
    # Suppression still consumes the one-time mint; the disclosure never shows up later.
    assert config.has_device_id() is True


def test_first_run_notice_not_minted_while_telemetry_inert(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["assembly", "doctor"])
    with telemetry.track("assembly doctor"):  # no token -> inert, nothing collected
        pass
    assert "Anonymous usage data" not in capsys.readouterr().err
    assert config.has_device_id() is False


@pytest.mark.parametrize("exc", [OSError("disk full"), CLIError("corrupt config")])
def test_first_run_notice_failures_never_break_the_command(events, monkeypatch, exc):
    def explode():
        raise exc

    monkeypatch.setattr(config, "has_device_id", explode)
    with telemetry.track("assembly doctor"):
        pass  # must not raise


@pytest.mark.parametrize(
    ("raw_args", "suppressed"),
    [
        (["--quiet"], True),
        (["-q", "doctor"], True),
        (["transcribe", "x.wav", "--json"], True),
        (["-j"], True),
        (["-o", "json"], True),
        (["-o", "json", "extra"], True),
        (["--output", "json"], True),
        (["--output=json"], True),
        (["-ojson"], True),
        (["-o", "text"], False),
        (["-o"], False),
        (["transcribe", "x.wav"], False),
        ([], False),
    ],
    ids=repr,
)
def test_notice_suppression_matches_quiet_and_json_forms(raw_args, suppressed):
    assert telemetry._notice_suppressed(raw_args) is suppressed
