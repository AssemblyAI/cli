import httpx
import pytest

from aai_cli.auth import ams
from aai_cli.errors import NotAuthenticated


def _patch_transport(monkeypatch, handler):
    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ams.httpx, "Client", fake_client)


def test_get_balance_sends_cookie_and_parses(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(200, json={"account_id": 1, "balance_in_cents": 2500})

    _patch_transport(monkeypatch, handler)
    out = ams.get_balance("jwt_1")
    assert out["balance_in_cents"] == 2500
    assert "/v2/billing/balance" in seen["url"]
    assert "stytch_session_jwt=jwt_1" in seen["cookie"]


def test_get_usage_posts_date_range(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"usage_items": []})

    _patch_transport(monkeypatch, handler)
    ams.get_usage("jwt", "2026-05-01", "2026-06-01", "day")
    assert "2026-05-01" in seen["body"] and "2026-06-01" in seen["body"]
    assert "day" in seen["body"]


def test_get_rate_limits_uses_account_path(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"rate_limits": []})

    _patch_transport(monkeypatch, handler)
    ams.get_rate_limits(42, "jwt")
    assert "/v1/users/accounts/42/rate-limits" in seen["url"]


def test_rename_token_puts_name_and_tolerates_empty_body(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.read().decode()
        return httpx.Response(200, content=b"")  # empty body, not JSON

    _patch_transport(monkeypatch, handler)
    ams.rename_token(42, 7, "prod", "jwt")  # must not raise
    assert seen["method"] == "PUT"
    assert "/v1/users/accounts/42/tokens/7" in seen["url"]
    assert "prod" in seen["body"]


def test_list_streaming_passes_filters(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"page_details": {}, "data": []})

    _patch_transport(monkeypatch, handler)
    ams.list_streaming("jwt", limit=5, status="completed")
    assert "limit=5" in seen["url"]
    assert "status=completed" in seen["url"]


def test_get_streaming_by_id(monkeypatch):
    def handler(request):
        assert "/v1/users/streaming/s_1" in str(request.url)
        return httpx.Response(200, json={"session_id": "s_1", "status": "completed"})

    _patch_transport(monkeypatch, handler)
    out = ams.get_streaming("s_1", "jwt")
    assert out["session_id"] == "s_1"


def test_account_4xx_raises_not_authenticated(monkeypatch):
    def handler(request):
        return httpx.Response(401, json={"detail": "expired"})

    _patch_transport(monkeypatch, handler)
    with pytest.raises(NotAuthenticated):
        ams.get_balance("jwt")
