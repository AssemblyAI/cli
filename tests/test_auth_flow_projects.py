"""`flow.find_or_create_cli_key` project-resolution tests.

Split out of test_auth_flow.py to keep that module under the 500-line gate. These
cover reusing an existing CLI key, minting into the first usable project, and the
clean APIErrors for the no-project / malformed-list shapes.
"""

import pytest

from aai_cli.auth import flow
from aai_cli.core.errors import APIError


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


def test_find_or_create_uses_first_entry_that_has_a_project(monkeypatch):
    # The first membership carries no project object; a usable project later in the
    # list must still be minted into rather than failing with "no project".
    projects = [
        {"tokens": []},
        {"project": {"id": 9}, "tokens": []},
    ]
    monkeypatch.setattr(flow.ams, "list_projects", lambda acct, jwt: projects)

    created = {}

    def fake_create(account_id, project_id, token_name, session_jwt):
        created.update(project_id=project_id)
        return {"api_key": "sk_later"}

    monkeypatch.setattr(flow.ams, "create_token", fake_create)
    assert flow.find_or_create_cli_key(1, "jwt") == "sk_later"
    assert created == {"project_id": 9}


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
