import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/voice-agent")


def _load_app(monkeypatch):
    for name in ("api.index", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))
    return importlib.import_module("api.index")


def _ok_response(token="tok-123"):
    resp = MagicMock()
    resp.json.return_value = {"token": token}
    resp.raise_for_status.return_value = None
    return resp


def test_token_returns_token_and_agent_ws_url(monkeypatch):
    mod = _load_app(monkeypatch)
    monkeypatch.setattr(mod.httpx2, "get", lambda *a, **k: _ok_response())
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "tok-123"
    assert body["ws_url"].startswith("wss://") and body["ws_url"].endswith("/v1/ws")


def test_token_uses_bearer_authorization_header(monkeypatch):
    # The Voice Agent token uses Bearer auth (unlike the streaming token).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-test")
    mod = _load_app(monkeypatch)
    captured = {}

    def fake_get(url, params=None, headers=None):
        captured.update(url=url, headers=headers)
        return _ok_response()

    monkeypatch.setattr(mod.httpx2, "get", fake_get)
    TestClient(mod.app).post("/api/token")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert "/v1/token" in captured["url"]


def test_page_reads_reply_audio_from_data_field():
    # reply.audio carries the base64 PCM in `data` (not `audio`); guard the regression.
    app_js = (TEMPLATE_DIR / "public" / "static" / "app.js").read_text()
    assert "reply.audio" in app_js
    assert "event.data" in app_js


def test_token_surfaces_error_as_502(monkeypatch):
    mod = _load_app(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod.httpx2, "get", boom)
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 502
