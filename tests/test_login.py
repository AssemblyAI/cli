import json

from typer.testing import CliRunner

from aai_cli.auth.flow import LoginResult
from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


def _login_result(key="sk_from_oauth"):
    return LoginResult(api_key=key, session_jwt="jwt_x", session_token="tok_x", account_id=7)


def _fake_login_result(*, json_mode=False):
    return _login_result()


def test_login_with_api_key_flag_stores_key(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag", "--json"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag"
    # CliRunner merges stderr into output, so the deprecation {"warning": …} line
    # rides along — filter by the key the success payload carries.
    objs = [json.loads(line) for line in result.output.strip().splitlines()]
    assert next(o for o in objs if "authenticated" in o)["authenticated"] is True
    warning = next(o for o in objs if "warning" in o)["warning"]
    assert "--with-api-key" in warning  # the deprecation points at the stdin form


def test_login_rejects_invalid_key(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=False)
    result = runner.invoke(app, ["login", "--api-key", "sk_bad"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None
    # is_auth_failure treats 401 *and* 403 as a rejected key (corporate proxies
    # often answer 403), so the message must not claim a specific 401.
    assert "rejected (HTTP 401/403)" in result.output


def test_login_stores_under_named_profile(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["--profile", "staging", "login", "--api-key", "sk_s"])
    assert result.exit_code == 0
    assert config.get_api_key("staging") == "sk_s"


def test_whoami_reports_authenticated(mocker):
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["profile"] == "default"
    assert data["reachable"] is True
    assert data["api_key"].startswith("sk_") and "…" in data["api_key"]


def test_whoami_human_render_shows_detail_rows(mocker):
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0
    # The shared borderless detail grid: labelled rows, no JSON, key masked.
    assert "Profile" in result.output and "default" in result.output
    assert "reachable" in result.output
    assert "…" in result.output and '"profile"' not in result.output


def test_whoami_unauthenticated_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _fake_login_result)
    validate = mocker.patch(
        "aai_cli.commands.login.client.validate_key", autospec=True, return_value=True
    )
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 4
    assert config.get_api_key("default") == "sk_from_oauth"
    validate.assert_not_called()
    assert "Run the same command again" in result.output


def test_logout_clears_key():
    config.set_api_key("default", "sk_1234567890")
    result = runner.invoke(app, ["logout", "--json"])
    assert result.exit_code == 0
    assert config.get_api_key("default") is None
    assert json.loads(result.output)["logged_out"] is True  # pins the success flag


def test_login_oauth_flow_stores_returned_key(monkeypatch):
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"


def test_login_oauth_persists_session(monkeypatch):
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert config.get_session("default") == {"jwt": "jwt_x", "token": "tok_x"}
    assert config.get_account_id("default") == 7


def test_login_api_key_flag_does_not_persist_session(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_session("default") is None


def test_login_api_key_flag_warns_account_commands_need_browser_login(mocker):
    # An --api-key login stores no browser session, so the user is told up front that
    # account self-service commands need `assembly login` without --api-key.
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    # (Substring chosen to survive Rich soft-wrapping of the longer hint line.)
    assert "Account commands (keys/balance/usage/limits/audit) need" in result.output


def test_login_browser_path_has_no_api_key_only_note(monkeypatch):
    # The browser login DOES create a session, so the api-key-only caveat must not show.
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert "Signed in as default" in result.output
    assert "assembly onboard" in result.output  # the shared next-step hint
    assert "without --api-key" not in result.output
    assert config.get_session("default") is not None


def test_login_api_key_flag_clears_prior_browser_session(mocker):
    # A profile previously authenticated via browser becomes api-key-only on
    # re-login; the stale session must not linger and silently authenticate
    # account self-service commands as the previous identity.
    config.set_session("default", session_jwt="old_j", session_token="old_t", account_id=5)
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert config.get_session("default") is None
    assert config.get_account_id("default") is None


def test_logout_clears_session():
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=7)
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 0
    assert config.get_session("default") is None


def test_login_oauth_flow_failure_exits_nonzero(monkeypatch):
    from aai_cli.core.errors import APIError

    def boom(**_kwargs):
        raise APIError("Login failed: the server returned an unexpected response.")

    monkeypatch.setattr("aai_cli.auth.run_login_flow", boom)
    result = runner.invoke(app, ["login"])
    assert result.exit_code != 0
    assert config.get_api_key("default") is None


def test_login_timeout_is_auth_typed_with_exit_4(monkeypatch):
    # The loopback timeout surfaces as not_authenticated/exit 4, not api_error/1.
    from aai_cli.core.errors import NotAuthenticated

    def timed_out(**_kwargs):
        raise NotAuthenticated("Login timed out waiting for the browser.")

    monkeypatch.setattr("aai_cli.auth.run_login_flow", timed_out)
    result = runner.invoke(app, ["login", "--json"])
    assert result.exit_code == 4
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "not_authenticated"
    assert config.get_api_key("default") is None


def test_login_empty_api_key_flag_is_usage_error(monkeypatch):
    # `--api-key "$UNSET_VAR"` (an explicit empty value) must not silently fall
    # into the browser flow.
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("empty --api-key must not start a browser")
        ),
    )
    result = runner.invoke(app, ["login", "--api-key", ""])
    assert result.exit_code == 2
    assert "empty" in result.output
    assert config.get_api_key("default") is None


def test_login_whitespace_api_key_flag_is_usage_error(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("blank --api-key must not start a browser")
        ),
    )
    result = runner.invoke(app, ["login", "--api-key", "   "])
    assert result.exit_code == 2
    assert config.get_api_key("default") is None


def test_login_empty_api_key_flag_json_error_shape():
    result = runner.invoke(app, ["login", "--api-key", "", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["error"]["type"] == "usage_error"
    assert "--api-key" in payload["error"]["message"]


def test_login_api_key_flag_still_bypasses_oauth(monkeypatch, mocker):
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("OAuth must not run with --api-key")),
    )
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag2"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_flag2"


def test_login_binds_env_to_profile(monkeypatch):
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _fake_login_result)
    result = runner.invoke(app, ["--env", "sandbox000", "login"])
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_from_oauth"
    assert config.get_profile_env("default") == "sandbox000"


def test_login_defaults_to_production_ignoring_stored_sandbox_env(monkeypatch):
    # A bare `assembly login` must default to production even when the profile was
    # previously bound to the sandbox — re-signing-in shouldn't silently re-target it.
    config.set_profile_env("default", "sandbox000")
    seen = {}

    def fake(*, json_mode=False):
        from aai_cli.core import environments

        # The browser OAuth/AMS flow reads environments.active(); capture what it sees.
        seen["active"] = environments.active().name
        return _login_result()

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fake)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 0
    assert seen["active"] == "production"  # the flow ran against prod, not the stored sandbox
    assert config.get_profile_env("default") == "production"  # and re-bound the profile to prod


def test_login_explicit_sandbox_still_overrides_production_default(monkeypatch):
    # The production default is only the fallback: an explicit --sandbox still wins, and
    # the flow runs against (and the profile binds to) the sandbox.
    config.set_profile_env("default", "production")
    seen = {}

    def fake(*, json_mode=False):
        from aai_cli.core import environments

        seen["active"] = environments.active().name
        return _login_result()

    monkeypatch.setattr("aai_cli.auth.run_login_flow", fake)
    result = runner.invoke(app, ["--sandbox", "login"])
    assert result.exit_code == 0
    assert seen["active"] == "sandbox000"
    assert config.get_profile_env("default") == "sandbox000"


def test_sandbox_flag_is_shortcut_for_env(monkeypatch):
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow", lambda *, json_mode=False: _login_result("sk_x")
    )
    result = runner.invoke(app, ["--sandbox", "login"])
    assert result.exit_code == 0
    assert config.get_profile_env("default") == "sandbox000"


def test_whoami_reports_env(mocker):
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["--env", "production", "whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["env"] == "production"


def test_root_callback_keeps_profile_env_without_sandbox(mocker):
    # Without --sandbox the profile's own env must stand (pins `sandbox and env is
    # None`: an `or` would force sandbox000 onto every default invocation).
    config.set_api_key("default", "sk_1234567890")
    config.set_profile_env("default", "production")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["env"] == "production"


def test_root_callback_sandbox_overrides_profile_env(mocker):
    # --sandbox forces sandbox000 even when the profile is bound elsewhere (pins the
    # `env is None` arm: an `is not None` would leave the profile env in place).
    config.set_api_key("default", "sk_1234567890")
    config.set_profile_env("default", "production")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["--sandbox", "whoami", "--json"])
    assert result.exit_code == 0
    # In --json mode the env-mismatch warning is its own {"warning": …} object, so pick
    # the whoami payload (the line carrying "env") rather than assuming an ordering.
    objs = [json.loads(line) for line in result.output.strip().splitlines()]
    payload = next(obj for obj in objs if "env" in obj)
    assert payload["env"] == "sandbox000"


def test_unknown_env_exits_2(mocker):
    # Routed through the standard error path. Output is human-by-default (the root
    # callback can't see a per-command --json, and we never auto-switch to JSON on a
    # pipe/agent), so it's the "Error:" + "Suggestion:" pair on stderr, not a JSON blob —
    # regardless of whether stdout is a TTY.
    is_agentic = mocker.patch("aai_cli.ui.output.is_agentic", autospec=True)
    for agentic in (True, False):
        is_agentic.return_value = agentic
        result = runner.invoke(app, ["--env", "bogus", "whoami"])
        assert result.exit_code == 2
        assert "Error:" in result.output
        assert "Suggestion:" in result.output
        assert "unset AAI_ENV" in result.output
        assert '"error"' not in result.output  # never the JSON shape


def test_root_callback_error_honors_json_request():
    # A callback-level failure (bad --env) runs before the subcommand parses its own
    # --json, but a `… --json` pipeline still expects the uniform {"error": …} shape, so
    # the callback sniffs the pending command line and emits JSON for every failure class.
    for args in (
        ["--env", "bogus", "whoami", "--json"],
        ["--env", "bogus", "transcripts", "list", "--json"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 2
        payload = json.loads(result.output)
        assert payload["error"]["type"] == "invalid_environment"


def test_env_override_prints_warning_to_stderr(mocker):
    # The root callback warns when an explicit --env contradicts the profile's stored
    # env (the stored key was minted for a different environment).
    config.set_api_key("default", "sk_1234567890")
    config.set_profile_env("default", "production")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["--env", "sandbox000", "whoami", "--json"])
    assert result.exit_code == 0
    assert "may be rejected by sandbox000" in result.output


def test_rejected_api_key_has_suggestion(monkeypatch):
    from aai_cli.core import client

    monkeypatch.setattr(client, "validate_key", lambda key: False)
    result = runner.invoke(app, ["login", "--api-key", "sk_bad", "--json"])
    assert result.exit_code == 1
    assert "Check the key and retry" in result.output


def test_whoami_reports_session_and_account(mocker):
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["account_id"] == 77
    assert data["session"] == "stored"


def test_whoami_session_none_without_browser_login(mocker):
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["session"] == "none"
    assert data["account_id"] is None


def test_whoami_renders_human_table_reachable(mocker):
    # Human-readable (non-JSON) render path: the grid lists profile, env, masked key,
    # a reachable status, and the account/session rows.
    config.set_api_key("default", "sk_1234567890")
    config.set_session("default", session_jwt="j", session_token="t", account_id=77)
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 0
    assert "Profile" in result.output
    assert "default" in result.output
    assert "reachable" in result.output
    assert "77" in result.output
    assert "stored" in result.output
    assert "sk_1234567890" not in result.output  # masked


def test_whoami_renders_human_table_rejected_key(mocker):
    # The non-JSON render path also covers the "key rejected" branch and the
    # account/session "none" fallbacks (the em-dash placeholder). A rejected key
    # is a failed preflight: the status still renders, but the exit code is 4.
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=False)
    result = runner.invoke(app, ["whoami"])
    assert result.exit_code == 4
    assert "key rejected" in result.output
    assert "none" in result.output


def test_whoami_rejected_key_exits_4_with_json_status_on_stdout(mocker):
    # CI preflight contract: a rejected key keeps the rendered status (stdout, clean
    # JSON) but signals failure via the auth exit code 4.
    config.set_api_key("default", "sk_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=False)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 4
    data = json.loads(result.output)
    assert data["reachable"] is False
    assert data["profile"] == "default"


def test_whoami_honors_env_api_key(monkeypatch, mocker):
    # A CI box authenticated only via ASSEMBLYAI_API_KEY (no keyring entry) must be
    # able to use whoami as a preflight check.
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_env_1234567890")
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["whoami", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["reachable"] is True
    assert "sk_env_1234567890" not in result.output  # still masked


def test_login_failure_never_auto_logs_in_again(monkeypatch):
    # The login command owns sign-in (auto_login=False): a NotAuthenticated from its
    # own browser flow must surface as exit 4, not trigger run_command's auto-login
    # retry — even in an interactive session.
    from aai_cli.core.errors import NotAuthenticated

    calls = {"n": 0}

    def timed_out(**_kwargs):
        calls["n"] += 1
        raise NotAuthenticated("Login timed out waiting for the browser.")

    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.auth.run_login_flow", timed_out)
    result = runner.invoke(app, ["login"])
    assert result.exit_code == 4
    assert calls["n"] == 1  # the command's own attempt only; no auto-login retry


def test_logout_never_auto_logs_in(monkeypatch):
    # Signing out must never start a sign-in flow (auto_login=False), even if the
    # body surfaces a NotAuthenticated and the session is interactive.
    from aai_cli.core.errors import NotAuthenticated

    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: True)
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda **_: (_ for _ in ()).throw(AssertionError("logout must never start a login")),
    )
    monkeypatch.setattr(
        "aai_cli.commands.login.config.clear_api_key",
        lambda _profile: (_ for _ in ()).throw(NotAuthenticated()),
    )
    result = runner.invoke(app, ["logout"])
    assert result.exit_code == 4


def test_invalid_profile_name_fails_before_any_network(mocker):
    # `assembly --profile 'bad name!' login --api-key X` used to validate the key over
    # the network first and only reject the profile at keyring-write time; the root
    # callback now rejects it at resolution time.
    validate = mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True)
    result = runner.invoke(app, ["--profile", "bad name!", "login", "--api-key", "sk_x"])
    assert result.exit_code == 2
    assert "Invalid profile name" in result.output
    validate.assert_not_called()
