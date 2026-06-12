import pytest

from aai_cli.auth import flow, loopback
from aai_cli.errors import APIError, NotAuthenticated


class _FakeCapture:
    """Stands in for an already-bound loopback server: wait() returns a canned result."""

    def __init__(self, result, log=None):
        self._result = result
        self._log = log

    def wait(self, timeout=120.0):
        if self._log is not None:
            self._log.append("wait")
        return self._result


def _fake_start_capture(monkeypatch, result):
    monkeypatch.setattr(flow, "_start_capture", lambda: _FakeCapture(result))


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


def test_start_capture_delegates_to_loopback(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(flow.loopback, "start_capture", lambda: sentinel)
    assert flow._start_capture() is sentinel


def test_run_login_flow_opens_the_discovery_start_url(monkeypatch):
    # The browser is opened with exactly the URL build_start_url() produces.
    seen = {}
    monkeypatch.setattr(flow.discovery, "build_start_url", lambda: "start-url")
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: seen.setdefault("url", url))
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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
        lambda ist, org: {"account": {"id": 9}, "session_jwt": "jwt", "session_token": "t"},
    )
    monkeypatch.setattr(flow, "find_or_create_cli_key", lambda acct, jwt: "sk_final")

    flow.run_login_flow()
    assert seen["url"] == "start-url"


def test_run_login_flow_rejects_wrong_token_type(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="something_else")
    )
    with pytest.raises(APIError) as exc:
        flow.run_login_flow()
    assert "valid OAuth token" in exc.value.message


def test_run_login_flow_happy_path(monkeypatch):
    opened = {}
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: opened.setdefault("url", url))
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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


def test_run_login_flow_timeout_raises_auth_typed_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(monkeypatch, loopback.CallbackResult(error="timeout"))
    with pytest.raises(NotAuthenticated) as exc:
        flow.run_login_flow()
    assert exc.value.message == "Login timed out waiting for the browser."
    assert exc.value.error_type == "not_authenticated"  # auth-typed, not api_error
    assert exc.value.exit_code == 4
    assert (
        exc.value.suggestion
        == "Run 'assembly login' again, or use 'assembly login --api-key <KEY>'."
    )


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
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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

    flow._open_browser("https://login.example", json_mode=False)

    err = capsys.readouterr().err
    assert "  https://login.example" in err
    assert "Could not open a browser" in err


def test_run_login_flow_missing_session_token_raises_api_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
    )
    monkeypatch.setattr(
        flow.ams,
        "discover",
        lambda token: {"organizations": [{"organization_id": "org_1"}]},  # no IST
    )
    with pytest.raises(APIError):
        flow.run_login_flow()


def test_run_login_flow_org_missing_id_raises_api_error(monkeypatch):
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch,
        loopback.CallbackResult(token="tok", token_type="discovery_oauth", error=None),
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


def _stub_ams_happy_path(monkeypatch):
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


def test_run_login_flow_binds_loopback_before_opening_browser(monkeypatch):
    # The callback server must be bound before the browser launches: a taken port
    # has to fail the flow before the user is mid-OAuth. wait() only happens after.
    order = []

    def fake_start():
        order.append("bind")
        return _FakeCapture(
            loopback.CallbackResult(token="tok", token_type="discovery_oauth"), log=order
        )

    monkeypatch.setattr(flow, "_start_capture", fake_start)
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: order.append("browser"))
    _stub_ams_happy_path(monkeypatch)

    assert flow.run_login_flow().api_key == "sk_final"
    assert order == ["bind", "browser", "wait"]


def test_run_login_flow_bind_failure_never_opens_browser(monkeypatch):
    def fail_start():
        raise APIError("Could not start the login callback server on 127.0.0.1:8123.")

    monkeypatch.setattr(flow, "_start_capture", fail_start)
    opened = []
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: opened.append(url))

    with pytest.raises(APIError, match="callback server"):
        flow.run_login_flow()
    assert opened == []  # the user is never sent into an OAuth flow that already failed


def test_run_login_flow_prints_waiting_hint(monkeypatch, capsys):
    # Headless/slow logins must not sit in 120s of silence: the flow says it is
    # waiting and names the non-browser alternative.
    monkeypatch.setattr(flow, "_open_browser", lambda url, **_: None)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
    )
    _stub_ams_happy_path(monkeypatch)

    assert flow.run_login_flow().api_key == "sk_final"
    err = capsys.readouterr().err
    assert "Waiting up to 2 minutes" in err
    assert "assembly login --api-key" in err
    assert '"hint"' not in err  # json_mode defaults to False: prose, not JSON objects


def test_open_browser_warns_when_open_returns_false(monkeypatch, capsys):
    # webbrowser.open returns False — without raising — on headless boxes; the user
    # must still get the manual-URL fallback rather than 120s of silence.
    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: False)
    flow._open_browser("https://login.example", json_mode=False)
    err = capsys.readouterr().err
    assert "Could not open a browser" in err
    assert "  https://login.example" in err


def test_open_browser_no_fallback_when_open_succeeds(monkeypatch, capsys):
    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: True)
    flow._open_browser("https://login.example", json_mode=False)
    err = capsys.readouterr().err
    assert "Opening your browser" in err
    assert "Could not open a browser" not in err


def test_open_browser_json_mode_emits_structured_hints(monkeypatch, capsys):
    import json

    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: False)
    flow._open_browser("https://login.example", json_mode=True)
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    assert len(lines) == 2  # every stderr line is machine-readable
    assert "browser" in lines[0]["hint"]
    assert lines[0]["url"] == "https://login.example"
    assert "Could not open a browser" in lines[1]["hint"]
    assert lines[1]["url"] == "https://login.example"


def test_run_login_flow_json_mode_keeps_stderr_machine_readable(monkeypatch, capsys):
    import json

    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: True)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
    )
    _stub_ams_happy_path(monkeypatch)

    assert flow.run_login_flow(json_mode=True).api_key == "sk_final"
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    waiting = next(obj for obj in lines if "Waiting up to 2 minutes" in obj["hint"])
    assert "assembly login --api-key" in waiting["hint"]
    assert "url" not in waiting  # the url field only ships on the browser-open notes
    opening = next(obj for obj in lines if "Opening your browser" in obj["hint"])
    assert opening["url"].startswith("https://")


def test_run_login_flow_multi_org_json_note(monkeypatch, capsys):
    import json

    monkeypatch.setattr(flow.webbrowser, "open", lambda _url: True)
    _fake_start_capture(
        monkeypatch, loopback.CallbackResult(token="tok", token_type="discovery_oauth")
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

    assert flow.run_login_flow(json_mode=True).api_key == "sk_final"
    lines = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    assert any("Acme" in obj["hint"] for obj in lines)  # the chosen org is still named
