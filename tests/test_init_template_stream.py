import importlib
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/live-captions")


def _load_app(monkeypatch):
    # Boot with a key so the endpoint's missing-key guard passes and reaches the mock;
    # preserve a key a test set before calling us (e.g. the SDK-client test).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", os.environ.get("ASSEMBLYAI_API_KEY") or "test-key")
    for name in ("api.index", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))
    return importlib.import_module("api.index")


def test_token_returns_token_and_streaming_ws_url(monkeypatch):
    mod = _load_app(monkeypatch)

    class FakeClient:
        def __init__(self, options):
            self.options = options

        def create_temporary_token(self, expires_in_seconds):
            return "tok-123"

    monkeypatch.setattr(mod, "StreamingClient", FakeClient)
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 200
    body = resp.json()
    assert body["token"] == "tok-123"
    assert body["ws_url"].startswith("wss://") and body["ws_url"].endswith("/v3/ws")


def test_token_builds_sdk_client_with_key_and_host(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-test")
    mod = _load_app(monkeypatch)
    captured = {}

    class FakeOptions:
        def __init__(self, *, api_key, api_host):
            captured["api_key"] = api_key
            captured["api_host"] = api_host

    class FakeClient:
        def __init__(self, options):
            pass

        def create_temporary_token(self, expires_in_seconds):
            captured["expires_in_seconds"] = expires_in_seconds
            return "tok-123"

    monkeypatch.setattr(mod, "StreamingClientOptions", FakeOptions)
    monkeypatch.setattr(mod, "StreamingClient", FakeClient)
    TestClient(mod.app).post("/api/token")
    assert captured["api_key"] == "sk-test"
    assert captured["api_host"] == mod.settings.STREAMING_HOST
    assert captured["expires_in_seconds"] == mod.settings.TOKEN_EXPIRES_IN_SECONDS


def test_token_surfaces_error_as_502(monkeypatch):
    mod = _load_app(monkeypatch)

    class FakeClient:
        def __init__(self, options):
            pass

        def create_temporary_token(self, expires_in_seconds):
            raise RuntimeError("network down")

    monkeypatch.setattr(mod, "StreamingClient", FakeClient)
    resp = TestClient(mod.app).post("/api/token")
    assert resp.status_code == 502


def test_index_route_serves_page(monkeypatch):
    mod = _load_app(monkeypatch)
    resp = TestClient(mod.app).get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()
