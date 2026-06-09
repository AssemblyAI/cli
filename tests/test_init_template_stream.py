import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/live-captions")


def _load_app(monkeypatch):
    # Boot with a key so the endpoint's missing-key guard passes and reaches the mock;
    # preserve a key a test set before calling us (e.g. the raw-auth-header test).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", os.environ.get("ASSEMBLYAI_API_KEY") or "test-key")
    for name in ("api.index", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))
    return importlib.import_module("api.index")


def _ok_response(mocker, token="tok-123"):
    resp = mocker.MagicMock()
    resp.json.return_value = {"token": token}
    resp.raise_for_status.return_value = None
    return resp


def test_token_returns_token_and_streaming_ws_url(monkeypatch, mocker):
    mod = _load_app(monkeypatch)
    monkeypatch.setattr(mod.httpx2, "get", lambda *a, **k: _ok_response(mocker))
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "tok-123"
    assert body["ws_url"].startswith("wss://") and body["ws_url"].endswith("/v3/ws")


def test_token_uses_raw_authorization_header(monkeypatch, mocker):
    # The streaming token uses the raw key as Authorization (NOT Bearer).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-test")
    mod = _load_app(monkeypatch)
    captured = {}

    def fake_get(url, params=None, headers=None):
        captured.update(url=url, headers=headers)
        return _ok_response(mocker)

    monkeypatch.setattr(mod.httpx2, "get", fake_get)
    TestClient(mod.app).post("/api/token")
    assert captured["headers"]["Authorization"] == "sk-test"
    assert "/v3/token" in captured["url"]


def test_token_surfaces_error_as_502(monkeypatch):
    mod = _load_app(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod.httpx2, "get", boom)
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 502


def test_index_route_serves_page(monkeypatch):
    mod = _load_app(monkeypatch)
    resp = TestClient(mod.app).get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()
