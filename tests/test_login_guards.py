"""Tests for login keyring preflight, truthful logout, and offline whoami.

Split out of test_login.py to keep modules under the 500-line gate.
"""

import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.main import app

runner = CliRunner()


def _fake_login_result(key="sk_from_oauth"):
    return LoginResult(api_key=key, session_jwt="jwt_x", session_token="tok_x", account_id=7)


def test_login_browser_flow_fails_fast_when_keyring_unusable(monkeypatch):
    # A user must not complete the whole browser OAuth dance only to fail on the
    # final keyring write: probe the keyring first and point at what works.
    monkeypatch.setattr("aai_cli.commands.login.config.keyring_usable", lambda: False)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("browser flow must not start")),
    )
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 2
    assert "keyring" in result.output
    assert "ASSEMBLYAI_API_KEY" in result.output


def test_login_api_key_flow_fails_fast_when_keyring_unusable(monkeypatch, mocker):
    # --api-key also stores to the keyring, so it gets the same preflight — before
    # the key-validation network call.
    monkeypatch.setattr("aai_cli.commands.login.config.keyring_usable", lambda: False)
    validate = mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_x"])
    assert result.exit_code == 2
    validate.assert_not_called()
    assert config.get_api_key("default") is None


def test_login_keyring_preflight_json_error_shape(monkeypatch):
    monkeypatch.setattr("aai_cli.commands.login.config.keyring_usable", lambda: False)
    result = runner.invoke(app, ["login", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "keyring_unusable"
    assert "ASSEMBLYAI_API_KEY" in payload["error"]["suggestion"]


def test_login_empty_api_key_error_wins_over_keyring_preflight(monkeypatch):
    # An explicit empty --api-key is a usage error in its own right; it must be
    # reported as such even when the keyring is also unusable.
    monkeypatch.setattr("aai_cli.commands.login.config.keyring_usable", lambda: False)
    result = runner.invoke(app, ["login", "--api-key", "", "--json"])
    assert result.exit_code == 2
    assert json.loads(result.output)["error"]["type"] == "usage_error"


def test_login_passes_json_mode_to_browser_flow(monkeypatch):
    # --json must reach the flow so its stderr progress notes ship as {"hint": ...}
    # objects (machine-readable stderr), not human prose.
    seen = {}

    def fake(*, json_mode):
        seen["json_mode"] = json_mode
        return _fake_login_result()

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fake)
    assert runner.invoke(app, ["login", "--json"]).exit_code == 0
    assert seen["json_mode"] is True
    assert runner.invoke(app, ["login"]).exit_code == 0
    assert seen["json_mode"] is False


def test_logout_fresh_machine_reports_nothing_to_clear():
    # Idempotent (exit 0), but truthful: nothing was stored, so don't claim a
    # sign-out happened.
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "No stored credentials" in result.output
    assert "nothing to clear" in result.output
    assert "Signed out" not in result.output


def test_logout_json_reports_cleared_false_when_nothing_stored():
    result = runner.invoke(app, ["logout", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["cleared"] is False
    assert payload["logged_out"] is True


def test_logout_key_only_reports_cleared_true():
    config.set_api_key("default", "sk_1234567890")
    result = runner.invoke(app, ["logout", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["cleared"] is True


def test_logout_session_only_reports_cleared_true():
    # A browser session with no API key still counts as stored credentials (pins
    # `had_key or had_session`: an `and` would claim nothing was cleared).
    config.set_session("default", session_jwt="j", session_token="t", account_id=7)
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert "Signed out of default" in result.output


def test_whoami_network_failure_still_renders_table(mocker):
    # A network failure must not suppress the local identity table; the status is
    # "unreachable (network error)" — distinct from "key rejected" — and the exit
    # code is 1 (api_error), keeping 4 reserved for a key the server refused.
    from aai_cli.errors import APIError

    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    mocker.patch(
        "aai_cli.commands.login.client.validate_key",
        autospec=True,
        side_effect=APIError("Network error contacting AssemblyAI: connection refused"),
    )
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 1
    assert "Profile" in result.output and "default" in result.output
    assert "unreachable (network error)" in result.output
    assert "key rejected" not in result.output
    assert "not rejected" in result.output  # the suggestion distinguishes it from a 401


def test_whoami_network_failure_json_reachable_is_null(mocker):
    from aai_cli.errors import APIError

    config.set_api_key("default", "sk_1234567890")
    mocker.patch(
        "aai_cli.commands.login.client.validate_key",
        autospec=True,
        side_effect=APIError("Network error contacting AssemblyAI: connection refused"),
    )
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 1
    objs = [json.loads(line) for line in result.output.strip().splitlines()]
    payload = next(o for o in objs if "profile" in o and "error" not in o)
    assert payload["reachable"] is None  # "couldn't check", not a rejection
    err = next(o for o in objs if "error" in o)
    assert err["error"]["type"] == "api_error"
