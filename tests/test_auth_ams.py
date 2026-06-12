import httpx2 as httpx
import pytest

from aai_cli.auth import ams
from aai_cli.errors import APIError, NotAuthenticated


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ams.httpx, "Client", fake_client)


def test_discover_posts_token_and_type(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "organizations": [{"organization_id": "org_1", "organization_name": "Acme"}],
                "email": "a@b.com",
                "intermediate_session_token": "ist_123",
            },
        )

    _patch_transport(monkeypatch, handler)
    out = ams.discover("tok_abc")
    assert out["intermediate_session_token"] == "ist_123"
    assert "/v2/auth/discover" in seen["url"]
    assert "discovery_oauth" in seen["body"]
    assert "tok_abc" in seen["body"]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_4xx_raises_not_authenticated(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "Invalid credentials"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated) as exc:
        ams.discover("bad")
    # A rejected AMS session now carries an actionable next step.
    assert exc.value.suggestion is not None and "assembly login" in exc.value.suggestion


def test_500_raises_api_error_with_detail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Something went wrong"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "Something went wrong" in str(exc.value)
    assert exc.value.suggestion is not None and "network" in exc.value.suggestion
    # The "detail" field is extracted, not the raw JSON body: the field name and its
    # braces must not leak (pins `mapping is not None and "detail" in mapping`).
    assert "detail" not in str(exc.value)


def test_ok_response_with_non_json_body_raises_clean_api_error(monkeypatch):
    # A 2xx whose body isn't JSON (proxy interference, truncation) must surface as
    # a clean AMS error, not escape as a raw JSONDecodeError that run_command can
    # only report as an internal bug.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy intercepted</html>")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "not valid JSON" in str(exc.value)
    assert "HTTP 200" in str(exc.value)
    assert exc.value.suggestion is not None and "support" in exc.value.suggestion


def test_error_with_non_json_body_falls_back_to_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream is down")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "upstream is down" in str(exc.value)


def test_error_with_empty_body_falls_back_to_status(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="")

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "HTTP 500" in str(exc.value)


def test_error_json_without_detail_falls_back_to_body_text(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "not the detail field"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "not the detail field" in str(exc.value)


def test_exchange_posts_ist_and_org(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"session_jwt": "jwt_1", "session_token": "st_1"})

    _patch_transport(monkeypatch, handler)
    out = ams.exchange("ist_abc", "org_42")
    assert out["session_jwt"] == "jwt_1"
    assert "/v2/auth/exchange" in seen["url"]
    assert "ist_abc" in seen["body"]
    assert "org_42" in seen["body"]


def test_list_projects_returns_list_and_sends_cookie(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json=[{"project": {"id": 1}, "tokens": []}])

    _patch_transport(monkeypatch, handler)
    out = ams.list_projects(7, "sess_jwt")
    assert isinstance(out, list)
    assert out[0]["project"] == {"id": 1}
    assert "/v1/users/accounts/7/projects" in seen["url"]
    assert "stytch_session_jwt=sess_jwt" in seen["cookie"]


def test_discover_rejects_non_object_json(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["bad"])

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError):
        ams.discover("token")


def test_list_projects_rejects_non_object_items(monkeypatch):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"project": {"id": 1}}, "bad"])

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError):
        ams.list_projects(7, "sess_jwt")


def test_create_token_posts_project_and_name(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"api_key": "key_xyz"})

    _patch_transport(monkeypatch, handler)
    out = ams.create_token(7, 9, "my-cli-token", "sess_jwt")
    assert out["api_key"] == "key_xyz"
    assert "/v1/users/accounts/7/tokens" in seen["url"]
    assert "my-cli-token" in seen["body"]
    assert "9" in seen["body"]
