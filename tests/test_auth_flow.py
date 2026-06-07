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


def test_find_or_create_rejects_malformed_project_list(monkeypatch):
    # A 200 with an unexpected shape (here: a project id that isn't an int) becomes a
    # clean "run login again" APIError rather than a traceback.
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: [{"project": {"id": "x"}}])
    with pytest.raises(APIError):
        flow.find_or_create_cli_key(1, "jwt")


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


def test_find_or_create_raises_when_first_project_has_no_project(monkeypatch):
    # A project entry that omits its "project" object can't be created into; surface a
    # clean APIError instead of crashing when reaching for the project id.
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: [{"tokens": []}])
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create"))
    with pytest.raises(APIError):
        flow.find_or_create_cli_key(1, "jwt")


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


def test_find_or_create_raises_when_no_projects(monkeypatch):
    # An account with zero projects can't hold a key; surface a clean APIError.
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: [])
    monkeypatch.setattr(flow.ams, "create_token", lambda *a, **k: pytest.fail("should not create"))
    with pytest.raises(APIError) as exc:
        flow.find_or_create_cli_key(1, "jwt")
    assert "no project" in exc.value.message


def test_capture_delegates_to_loopback(monkeypatch):
    sentinel = loopback.CallbackResult(token="tok", token_type="discovery_oauth")
    captured = {}

    def fake_capture(state):
        captured["state"] = state
        return sentinel

    monkeypatch.setattr(flow.loopback, "capture_callback", fake_capture)
    assert flow._capture("nonce-1") is sentinel
    assert captured["state"] == "nonce-1"  # the nonce is forwarded to the loopback


def test_run_login_flow_binds_state_nonce(monkeypatch):
    # The nonce build_start_url() carries to Stytch must be the same one the loopback
    # capture is told to expect, or a genuine callback would never be accepted.
    seen = {}
    monkeypatch.setattr(
        flow.discovery, "build_start_url", lambda state: seen.setdefault("url_state", state) or "u"
    )
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)

    def fake_capture(state):
        seen["capture_state"] = state
        return loopback.CallbackResult(token="tok", token_type="discovery_oauth")

    monkeypatch.setattr(flow, "_capture", fake_capture)
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
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    flow.run_login_flow()
    assert seen["url_state"] == seen["capture_state"]
    assert len(seen["capture_state"]) >= 32  # token_urlsafe(32) -> unguessable nonce


def test_run_login_flow_rejects_wrong_token_type(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="something_else"),
    )
    with pytest.raises(APIError) as exc:
        flow.run_login_flow()
    assert "valid OAuth token" in exc.value.message


def test_run_login_flow_happy_path(monkeypatch):
    opened = {}
    monkeypatch.setattr(flow, "_open_browser", lambda url: opened.setdefault("url", url))
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    assert flow.run_login_flow().api_key == "sk_final"
    assert opened["url"].startswith("https://")


def test_run_login_flow_timeout_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(flow, "_capture", lambda _state: loopback.CallbackResult(error="timeout"))
    with pytest.raises(APIError) as exc:
        flow.run_login_flow()
    assert exc.value.message == "Login timed out waiting for the browser."
    assert exc.value.suggestion == "Run 'aai login' again."


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


def test_run_login_flow_uses_exchange_account(monkeypatch):
    # The signed-in account comes from exchange()'s response; the flow must not make a
    # second round-trip to fetch it.
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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
        lambda ist, org: {"account": {"id": 42}, "session_jwt": "jwt", "session_token": "t"},
    )
    captured = {}

    def fake_find(acct, jwt):
        captured["acct"] = acct
        return "sk_final"

    monkeypatch.setattr(flow, "find_or_create_cli_key", fake_find)
    assert flow.run_login_flow().api_key == "sk_final"
    assert captured["acct"] == 42


def test_run_login_flow_multi_org_notes_selection(monkeypatch, capsys):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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
        flow.ams,
        "exchange",
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    assert flow.run_login_flow().api_key == "sk_final"
    err = capsys.readouterr().err
    assert "Acme" in err  # the chosen org is named rather than silently picked


def test_open_browser_prints_fallback_to_stderr(monkeypatch, capsys):
    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: (_ for _ in ()).throw(OSError()))

    flow._open_browser("https://login.example")

    err = capsys.readouterr().err
    assert "https://login.example" in err
    assert "Could not open a browser" in err


def test_run_login_flow_missing_session_token_raises_api_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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


def test_run_login_flow_zero_orgs_raises(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(token="tok", token_type="discovery_oauth"),
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


def test_run_login_flow_returns_session_material(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url: None)
    monkeypatch.setattr(
        flow,
        "_capture",
        lambda _state: loopback.CallbackResult(
            token="tok", token_type="discovery_oauth", error=None
        ),
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {
            "organizations": [{"organization_id": "org_1"}],
            "intermediate_session_token": "ist_1",
        },
    )
    monkeypatch.setattr(
        flow.ams,
        "exchange",
        lambda ist, org: {
            "session_jwt": "jwt_1",
            "session_token": "tok_1",
            "account": {"id": 99},
        },
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_key")

    result = flow.run_login_flow()
    assert result.api_key == "sk_key"
    assert result.session_jwt == "jwt_1"
    assert result.session_token == "tok_1"
    assert result.account_id == 99
