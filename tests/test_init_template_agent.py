import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/agent")


def _load_app():
    spec = importlib.util.spec_from_file_location("_tmpl_agent", TEMPLATE_DIR / "api" / "index.py")
    assert spec and spec.loader  # spec_from_file_location is typed Optional
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ok_response(token="tok-123"):
    resp = MagicMock()
    resp.json.return_value = {"token": token}
    resp.raise_for_status.return_value = None
    return resp


def test_token_returns_token_and_agent_ws_url(monkeypatch):
    mod = _load_app()
    monkeypatch.setattr(mod.httpx, "get", lambda *a, **k: _ok_response())
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "tok-123"
    assert body["ws_url"].startswith("wss://") and body["ws_url"].endswith("/v1/ws")


def test_token_uses_bearer_authorization_header(monkeypatch):
    # The Voice Agent token uses Bearer auth (unlike the streaming token).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-test")
    mod = _load_app()
    captured = {}

    def fake_get(url, params=None, headers=None):
        captured.update(url=url, headers=headers)
        return _ok_response()

    monkeypatch.setattr(mod.httpx, "get", fake_get)
    TestClient(mod.app).post("/api/token")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert "/v1/token" in captured["url"]


def test_page_reads_reply_audio_from_data_field():
    # reply.audio carries the base64 PCM in `data` (not `audio`); guard the regression.
    html = (TEMPLATE_DIR / "index.html").read_text()
    assert "reply.audio" in html
    assert "ev.data" in html


def test_token_surfaces_error_as_502(monkeypatch):
    mod = _load_app()

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod.httpx, "get", boom)
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 502
