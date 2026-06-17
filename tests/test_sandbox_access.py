"""The sandbox is restricted to AssemblyAI logins.

Two halves of the gate: the root callback rejects an internal-only environment for an
external account (and exempts `login` so an employee can bootstrap), and the root help
hides the sandbox flags/commands from external accounts. Identity is the login email
captured at browser login (`access.profile_is_internal`), so an API-key-only profile
reads as external.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aai_cli.auth import flow
from aai_cli.auth.flow import LoginResult
from aai_cli.auth.loopback import CallbackResult
from aai_cli.core import access, config
from aai_cli.main import _is_sandbox_command, app

runner = CliRunner()


# --- the internal-email predicate ------------------------------------------------


@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("alex@assemblyai.com", True),
        ("Alex@AssemblyAI.COM", True),  # case-insensitive
        ("  alex@assemblyai.com  ", True),  # whitespace-tolerant
        ("alex@other.com", False),
        ("alex@evil-assemblyai.com", False),  # the @ anchors the domain boundary
        ("alex@mail.assemblyai.com", False),  # a subdomain is not the org domain
        ("", False),
        (None, False),
    ],
)
def test_is_internal_email(email, expected):
    assert access.is_internal_email(email) is expected


def test_profile_email_roundtrips_and_drives_internal_check():
    config.set_api_key("default", "sk_x")
    config.set_profile_email("default", "alex@assemblyai.com")
    assert config.get_profile_email("default") == "alex@assemblyai.com"
    assert config.get_profile_email("nope") is None
    assert access.profile_is_internal("default") is True


def test_profile_is_internal_reads_the_active_profile_when_unspecified():
    assert access.profile_is_internal() is False  # empty config: no email
    config.set_profile_email(config.DEFAULT_PROFILE, "alex@assemblyai.com")
    assert access.profile_is_internal() is True


def test_profile_is_internal_fails_closed_on_corrupt_config(tmp_config):
    # A broken config.toml must read as external (and never crash --help), so the gate
    # can only ever *deny*, never accidentally grant, on a config it can't parse.
    (tmp_config / "config.toml").write_text("not = valid = toml", encoding="utf-8")
    assert access.profile_is_internal() is False


def test_persist_login_stores_email_atomically():
    config.persist_login(
        "default",
        api_key="sk_x",
        env="production",
        session_jwt="j",
        session_token="t",
        account_id=5,
        email="alex@assemblyai.com",
    )
    assert config.get_profile_email("default") == "alex@assemblyai.com"


def test_persist_login_without_email_leaves_it_unset():
    config.persist_login(
        "default",
        api_key="sk_x",
        env="production",
        session_jwt="j",
        session_token="t",
        account_id=5,
    )
    assert config.get_profile_email("default") is None


# --- the login flow captures the email ------------------------------------------


class _FakeCapture:
    def __init__(self, result):
        self._result = result

    def wait(self):
        return self._result


def _drive_login_flow(monkeypatch, discover_payload):
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    monkeypatch.setattr(
        flow,
        "_start_capture",
        lambda: _FakeCapture(CallbackResult(token="tok", token_type="discovery_oauth")),
    )
    monkeypatch.setattr(flow.ams, "discover", lambda token: discover_payload)
    monkeypatch.setattr(
        flow.ams,
        "exchange",
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")
    return flow.run_login_flow()


def test_run_login_flow_threads_discover_email_into_the_result(monkeypatch):
    result = _drive_login_flow(
        monkeypatch,
        {
            "organizations": [{"organization_id": "org_1"}],
            "email": "sam@assemblyai.com",
            "intermediate_session_token": "ist",
        },
    )
    assert result.email == "sam@assemblyai.com"


def test_run_login_flow_leaves_email_none_when_discover_omits_it(monkeypatch):
    result = _drive_login_flow(
        monkeypatch,
        {
            "organizations": [{"organization_id": "org_1"}],
            "intermediate_session_token": "ist",
        },
    )
    assert result.email is None


# --- the sandbox-command marker --------------------------------------------------


class _FakeCommand:
    def __init__(self, help_text=None, short_help=None):
        self.help = help_text
        self.short_help = short_help


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (_FakeCommand(help_text=r"\[sandbox] Do a thing"), True),  # docstring escapes the bracket
        (_FakeCommand(help_text="[sandbox] Do a thing"), True),  # already unescaped
        (_FakeCommand(help_text="   \\[sandbox] Indented"), True),  # leading whitespace
        (_FakeCommand(short_help=r"\[sandbox] Fallback"), True),  # short_help fallback
        (_FakeCommand(help_text="Transcribe a file"), False),
        (_FakeCommand(), False),
    ],
)
def test_is_sandbox_command(command, expected):
    assert _is_sandbox_command(command) is expected


# --- the root-callback gate ------------------------------------------------------


def test_external_account_is_rejected_for_the_sandbox_flag():
    result = runner.invoke(app, ["--sandbox", "transcripts", "list"])
    assert result.exit_code == 2
    assert "sandbox000 environment is restricted to AssemblyAI accounts" in result.output
    assert "assembly login" in result.output


def test_external_account_is_rejected_for_env_sandbox():
    # The --env path reaches the same gate (not just the --sandbox shortcut).
    result = runner.invoke(app, ["--env", "sandbox000", "transcripts", "list"])
    assert result.exit_code == 2
    assert "restricted to AssemblyAI accounts" in result.output


def test_rejection_uses_the_structured_error_envelope_in_json_mode():
    result = runner.invoke(app, ["--sandbox", "transcripts", "list", "--json"])
    assert result.exit_code == 2
    err = next(json.loads(line) for line in result.output.strip().splitlines() if "error" in line)
    assert err["error"]["type"] == "restricted_environment"


def test_production_is_never_gated_for_external_accounts():
    # The default environment is open to everyone — selecting it must not trip the gate
    # (the command fails later for lack of a key, not with the restricted-env error).
    result = runner.invoke(app, ["--env", "production", "transcripts", "list"])
    assert "restricted to AssemblyAI accounts" not in result.output


def test_login_is_exempt_so_an_employee_can_bootstrap_the_sandbox(monkeypatch):
    # A first-time employee has no stored email yet, so the gate would otherwise block
    # the very `login --sandbox` that records it. login is exempt; the email then lands.
    monkeypatch.setattr(
        "aai_cli.auth.run_login_flow",
        lambda *, json_mode=False: LoginResult(
            api_key="sk_x",
            session_jwt="j",
            session_token="t",
            account_id=1,
            email="dana@assemblyai.com",
        ),
    )
    result = runner.invoke(app, ["--sandbox", "login"])
    assert result.exit_code == 0
    assert "restricted to AssemblyAI accounts" not in result.output
    assert config.get_profile_email("default") == "dana@assemblyai.com"
    assert access.profile_is_internal("default") is True


@pytest.mark.usefixtures("internal_profile")
def test_internal_account_may_select_the_sandbox():
    from aai_cli.core import environments

    result = runner.invoke(app, ["--sandbox"])
    assert result.exit_code == 0
    assert environments.active().name == "sandbox000"


# --- help filtering --------------------------------------------------------------


def test_help_hides_the_sandbox_surface_from_external_accounts_and_restores_it(monkeypatch):
    external = runner.invoke(app, ["--help"])
    assert external.exit_code == 0
    # Both the flags and the [sandbox]-tagged commands are gone for an external account.
    assert "--sandbox" not in external.output
    assert "--env" not in external.output
    assert "[sandbox]" not in external.output
    assert "agent-cascade" not in external.output
    # …but the filter is surgical: non-sandbox flags and commands stay visible (this
    # also kills the mutant that would treat every option/command as sandbox).
    assert "--profile" in external.output
    assert "transcribe" in external.output

    # Internal accounts see the full surface — and this second render proves the
    # external one *restored* the hidden flags/commands rather than leaking hidden=True
    # onto the process-global Typer tree (which would hide them here too).
    monkeypatch.setattr("aai_cli.core.access.profile_is_internal", lambda *a, **k: True)
    internal = runner.invoke(app, ["--help"])
    assert "--sandbox" in internal.output
    assert "--env" in internal.output
    assert "[sandbox]" in internal.output
    assert "agent-cascade" in internal.output
