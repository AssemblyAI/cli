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
    monkeypatch.setattr(
        flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create")
    )
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
            "tokens": [
                {"name": "AssemblyAI CLI", "api_key": "sk_old", "is_disabled": True}
            ],
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
        lambda token: {"organizations": [], "email": "a@b.com", "intermediate_session_token": "ist"},
    )
    with pytest.raises(APIError, match="no AssemblyAI organization"):
        flow.run_login_flow()
