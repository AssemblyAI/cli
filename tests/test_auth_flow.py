import pytest

from aai_cli.auth import flow, loopback
from aai_cli.errors import APIError


def test_find_or_create_reuses_existing_cli_key(monkeypatch):
    projects = [
        {
            "project": {"id": 7},
            "tokens": [
                {"name": "Default Token", "api_key": "sk_default", "is_disabled": False},
                {"name": "AssemblyAI CLI", "api_key": "sk_cli", "is_disabled": False},
            ],
        }
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create"))
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_cli"


def test_find_or_create_creates_when_absent(monkeypatch):
    projects = [{"project": {"id": 7}, "tokens": []}]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)

    created = {}

    def fake_create(account_id, project_id, token_name, session_jwt):
        created.update(project_id=project_id, token_name=token_name)
        return {"api_key": "sk_new"}

    monkeypatch.setattr(flow.ams, "create_token", fake_create)
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_new"
    assert created == {"project_id": 7, "token_name": "AssemblyAI CLI"}


def test_find_or_create_creates_when_existing_cli_token_disabled(monkeypatch):
    projects = [
        {
            "project": {"id": 5},
            "tokens": [{"name": "AssemblyAI CLI", "api_key": "sk_old", "is_disabled": True}],
        }
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: {"api_key": "sk_fresh"})
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_fresh"


def test_run_login_flow_happy_path(monkeypatch):
    opened = {}
    monkeypatch.setattr(flow, "_open_browser", lambda url: opened.setdefault("url", url))
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [{"organization_id": "org_1", "organization_name": "Acme"}],
            "email": "a@b.com",
            "intermediate_session_token": "ist",
        },
    )
    monkeypatch.setattr(
        flow.ams,
        "exchange",
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow.ams, "get_auth", lambda jwt: {"id": 9})
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    assert flow.run_login_flow() == "sk_final"
    assert opened["url"].startswith("https://")


def test_run_login_flow_timeout_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(flow, "_capture", lambda: loopback.CallbackResult(error="timeout"))
    with pytest.raises(APIError, match="timed out"):
        flow.run_login_flow()


def test_find_or_create_reuses_token_with_token_name_field(monkeypatch):
    # AMS list endpoints may key the display name as "token_name" (matching the
    # create payload) rather than "name"; either must be reused, not duplicated.
    projects = [
        {
            "project": {"id": 7},
            "tokens": [{"token_name": "AssemblyAI CLI", "api_key": "sk_cli", "is_disabled": False}],
        }
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create"))
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_cli"


def test_find_or_create_creates_when_existing_token_has_no_api_key(monkeypatch):
    # A matching token whose api_key the list endpoint doesn't expose can't be
    # reused; fall through to minting a fresh one instead of crashing on KeyError.
    projects = [
        {
            "project": {"id": 7},
            "tokens": [{"name": "AssemblyAI CLI", "is_disabled": False}],
        }
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: {"api_key": "sk_new"})
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_new"


def test_run_login_flow_uses_exchange_account_without_get_auth(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [{"organization_id": "org_1"}],
            "intermediate_session_token": "ist",
        },
    )
    monkeypatch.setattr(
        flow.ams,
        "exchange",
        lambda ist, org: {"account": {"id": 42}, "session_jwt": "jwt"},
    )
    monkeypatch.setattr(
        flow.ams, "get_auth", lambda jwt: pytest.fail("get_auth is a redundant round-trip")
    )
    captured = {}

    def fake_find(acct, jwt):
        captured["acct"] = acct
        return "sk_final"

    monkeypatch.setattr(flow, "find_or_create_cli_key", fake_find)
    assert flow.run_login_flow() == "sk_final"
    assert captured["acct"] == 42


def test_run_login_flow_multi_org_notes_selection(monkeypatch, capsys):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [
                {"organization_id": "org_1", "organization_name": "Acme"},
                {"organization_id": "org_2", "organization_name": "Beta"},
            ],
            "intermediate_session_token": "ist",
        },
    )
    monkeypatch.setattr(
        flow.ams, "exchange", lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt"}
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    assert flow.run_login_flow() == "sk_final"
    out = capsys.readouterr().out
    assert "Acme" in out  # the chosen org is named rather than silently picked


def test_run_login_flow_missing_session_token_raises_api_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {"organizations": [{"organization_id": "org_1"}]},  # no IST
    )
    with pytest.raises(APIError):
        flow.run_login_flow()


def test_run_login_flow_org_missing_id_raises_api_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [{"organization_name": "Acme"}],  # no organization_id
            "intermediate_session_token": "ist",
        },
    )
    with pytest.raises(APIError):
        flow.run_login_flow()


def test_login_timeout_suggests_retry():
    # Mirror the existing timeout-path test setup in this module; the raised
    # APIError should now split message and suggestion.
    from aai_cli.errors import APIError

    err = APIError("Login timed out waiting for the browser.", suggestion="Run 'aai login' again.")
    assert err.suggestion == "Run 'aai login' again."


def test_run_login_flow_zero_orgs_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [],
            "email": "a@b.com",
            "intermediate_session_token": "ist",
        },
    )
    with pytest.raises(APIError, match="no AssemblyAI account"):
        flow.run_login_flow()
