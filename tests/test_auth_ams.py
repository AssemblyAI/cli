import httpx
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


def test_get_auth_sends_session_cookie(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json={"id": 42, "email": "a@b.com"})

    _patch_transport(monkeypatch, handler)
    out = ams.get_auth("sess_jwt_xyz")
    assert out["id"] == 42
    assert "stytch_session_jwt=sess_jwt_xyz" in seen["cookie"]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_4xx_raises_not_authenticated(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "Invalid credentials"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated):
        ams.discover("bad")


def test_500_raises_api_error_with_detail(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Something went wrong"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(APIError) as exc:
        ams.discover("x")
    assert "Something went wrong" in str(exc.value)
