"""Boot each ``assembly init`` template's FastAPI app in-process and drive every route.

The static contract gate (``scripts/template_contract_gate.py``) boots the real
Procfile process and curls ``GET /`` — but only ``/``. The ``install`` test resolves
requirements and imports the module. Neither exercises the ``/api/*`` surface, so a
route that imports fine but raises on every request would pass both.

This test closes that gap: it imports ``api.index`` (deps are in the dev group, so no
network/venv) and hits each route with ``TestClient`` for *both* outcomes —

  * the outbound AssemblyAI / LLM call **succeeds** -> assert the documented 200 body
  * the outbound call **raises** -> assert the graceful ``502 {"detail": ...}`` the
    templates promise ("clean 502, not a 500")

``raise_server_exceptions=False`` turns an *unhandled* exception into a 500 *response*
we can assert against, rather than letting it bubble out of the route.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

TEMPLATES_ROOT = Path("aai_cli/init/templates")
TEMPLATE_NAMES = sorted(
    p.name for p in TEMPLATES_ROOT.iterdir() if p.is_dir() and not p.name.startswith("__")
)
# live-captions now mints via the AssemblyAI SDK (covered in test_init_template_stream.py),
# so it no longer has the httpx2 GET these parametrized tests mock.
TOKEN_TEMPLATES = ["voice-agent"]
HTTP_OK = 200
HTTP_BAD_GATEWAY = 502


def _boom(*_args: object, **_kwargs: object) -> object:
    """Stand in for an outbound call that fails (missing key, network, bad plan)."""
    raise RuntimeError("simulated backend failure")


@contextmanager
def serve(template: str) -> Iterator[tuple[ModuleType, TestClient]]:
    """Import a template's ``api.index`` in isolation and yield it with a TestClient.

    The three templates ship an identically-named ``api`` package, so evict any cached
    ``api``/``api.*`` before and after to keep imports collision-free and order-independent
    (safe under pytest-xdist / pytest-randomly). The app reads ``ASSEMBLYAI_API_KEY`` at
    import; the autouse ``isolate_env`` fixture strips it, so we inject a dummy key before
    import so endpoints clear the ``_require_key`` guard and reach their mocked backends.
    (Tests that want to exercise the missing-key guard clear ``module.settings.API_KEY``
    after import — the guard re-reads it at request time.)
    """
    path = (TEMPLATES_ROOT / template).resolve()
    saved_path = list(sys.path)
    saved_modules = {n: m for n, m in sys.modules.items() if n == "api" or n.startswith("api.")}
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(path))
    saved_key = os.environ.get("ASSEMBLYAI_API_KEY")
    os.environ["ASSEMBLYAI_API_KEY"] = "test-key"
    try:
        module = importlib.import_module("api.index")
        client = TestClient(module.app, raise_server_exceptions=False)
        yield module, client
    finally:
        if saved_key is None:
            os.environ.pop("ASSEMBLYAI_API_KEY", None)
        else:
            os.environ["ASSEMBLYAI_API_KEY"] = saved_key
        for name in [n for n in sys.modules if n == "api" or n.startswith("api.")]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


@pytest.mark.parametrize("template", TEMPLATE_NAMES)
def test_serves_root_and_static_assets(template: str) -> None:
    with serve(template) as (_module, client):
        root = client.get("/")
        assert root.status_code == HTTP_OK
        assert "text/html" in root.headers["content-type"]
        assert root.text.strip()

        static_dir = TEMPLATES_ROOT / template / "static"
        assets = sorted(p for p in static_dir.glob("*") if p.is_file())
        assert assets, f"{template}: no static assets to serve"
        for asset in assets:
            resp = client.get(f"/static/{asset.name}")
            assert resp.status_code == HTTP_OK, f"{template}: GET /static/{asset.name}"


# --- audio-transcription: transcribe / status / ask --------------------------------


def test_app_applies_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # `assembly init` writes ASSEMBLYAI_BASE_URL when the key was minted for a non-prod
    # environment; the app must point the SDK at it. isolate_env strips it by default,
    # so set it here to exercise the import-time branch that applies it.
    monkeypatch.setenv("ASSEMBLYAI_BASE_URL", "https://api.example.test")
    with serve("audio-transcription") as (module, _client):
        assert module.aai.settings.base_url == "https://api.example.test"


def _fake_transcriber(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, transcript: object, *, raises: bool = False
) -> None:
    def submit(_audio: object, config: object = None) -> object:
        if raises:
            return _boom()
        return transcript

    monkeypatch.setattr(module.aai, "Transcriber", lambda: SimpleNamespace(submit=submit))


def test_transcribe_url_returns_submitted_id(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_transcriber(monkeypatch, module, SimpleNamespace(id="t-1"))
        resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
        assert resp.status_code == HTTP_OK
        assert resp.json() == {"id": "t-1"}


def test_transcribe_file_upload_returns_id(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_transcriber(monkeypatch, module, SimpleNamespace(id="t-2"))
        resp = client.post("/api/transcribe", files={"file": ("a.wav", b"RIFFfake", "audio/wav")})
        assert resp.status_code == HTTP_OK
        assert resp.json() == {"id": "t-2"}


def test_transcribe_without_id_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_transcriber(monkeypatch, module, SimpleNamespace(id=None))
        resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
        assert resp.status_code == HTTP_BAD_GATEWAY
        assert "detail" in resp.json()


def test_transcribe_failure_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_transcriber(monkeypatch, module, None, raises=True)
        resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
        assert resp.status_code == HTTP_BAD_GATEWAY
        assert "detail" in resp.json()


def _fake_get_transcript(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, result: object, *, raises: bool = False
) -> None:
    monkeypatch.setattr(
        module.Client,
        "get_default",
        staticmethod(lambda: SimpleNamespace(http_client=None)),
    )

    def fetch(_http: object, _tid: object) -> object:
        if raises:
            return _boom()
        return result

    monkeypatch.setattr(module, "get_transcript", fetch)


def test_status_completed_returns_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        status = module.aai.TranscriptStatus.completed
        done = SimpleNamespace(status=status, dict=lambda: {"id": "t", "text": "hi"})
        _fake_get_transcript(monkeypatch, module, done)
        resp = client.get("/api/status/t")
        assert resp.status_code == HTTP_OK
        assert resp.json() == {"status": "completed", "transcript": {"id": "t", "text": "hi"}}


def test_status_still_processing_reports_state(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        status = module.aai.TranscriptStatus.processing
        _fake_get_transcript(monkeypatch, module, SimpleNamespace(status=status))
        resp = client.get("/api/status/t")
        assert resp.status_code == HTTP_OK
        assert resp.json() == {"status": "processing"}


def test_status_error_is_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        status = module.aai.TranscriptStatus.error
        _fake_get_transcript(monkeypatch, module, SimpleNamespace(status=status, error="boom"))
        resp = client.get("/api/status/t")
        assert resp.status_code == HTTP_BAD_GATEWAY
        assert resp.json()["detail"] == "boom"


def test_status_fetch_failure_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_get_transcript(monkeypatch, module, None, raises=True)
        resp = client.get("/api/status/t")
        assert resp.status_code == HTTP_BAD_GATEWAY
        assert "detail" in resp.json()


def _fake_openai(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, reply: object, *, raises: bool = False
) -> None:
    def create(**_kwargs: object) -> object:
        if raises:
            return _boom()
        return reply

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    def make_client(**_kwargs: object) -> object:
        return fake

    monkeypatch.setattr(module, "OpenAI", make_client)


def test_ask_returns_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])
        _fake_openai(monkeypatch, module, reply)
        resp = client.post("/api/ask", json={"transcript_id": "t", "question": "q?"})
        assert resp.status_code == HTTP_OK
        assert resp.json() == {"answer": "yes"}


def test_ask_failure_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    with serve("audio-transcription") as (module, client):
        _fake_openai(monkeypatch, module, None, raises=True)
        resp = client.post("/api/ask", json={"transcript_id": "t", "question": "q?"})
        assert resp.status_code == HTTP_BAD_GATEWAY
        assert "detail" in resp.json()


# --- live-captions / voice-agent: /api/token ---------------------------------------


@pytest.mark.parametrize("template", TOKEN_TEMPLATES)
def test_token_mints_token_and_ws_url(template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, *, params: object, headers: object) -> object:
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"token": "tok-1"})

    with serve(template) as (module, client):
        monkeypatch.setattr(module.httpx2, "get", fake_get)
        result = client.post("/api/token")
        assert result.status_code == HTTP_OK
        body = result.json()
        assert body["token"] == "tok-1"
        assert body["ws_url"].startswith("wss://")


@pytest.mark.parametrize("template", TOKEN_TEMPLATES)
def test_token_failure_is_graceful(template: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with serve(template) as (module, client):
        monkeypatch.setattr(module.httpx2, "get", _boom)
        result = client.post("/api/token")
        assert result.status_code == HTTP_BAD_GATEWAY
        assert "detail" in result.json()


# --- missing API key: every template fails fast with an actionable message ----------

HTTP_INTERNAL_ERROR = 500

# (template, method, path, json body) for one representative key-using endpoint each.
MISSING_KEY_CASES = [
    ("voice-agent", "post", "/api/token", None),
    ("live-captions", "post", "/api/token", None),
    ("audio-transcription", "post", "/api/transcribe-url", {"url": "https://example.com/a.mp3"}),
]


@pytest.mark.parametrize(("template", "method", "path", "body"), MISSING_KEY_CASES)
def test_missing_key_returns_clear_error(
    template: str,
    method: str,
    path: str,
    body: dict[str, str] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset key yields a clear 500 (not a cryptic Bearer/SDK failure) before any call."""
    with serve(template) as (module, client):
        # serve() injects a dummy key at import; clear it so _require_key() trips at
        # request time (the guard re-reads settings.API_KEY on each request).
        monkeypatch.setattr(module.settings, "API_KEY", "")
        resp = client.request(method, path, json=body)
        assert resp.status_code == HTTP_INTERNAL_ERROR
        assert "ASSEMBLYAI_API_KEY is not set" in resp.json()["detail"]
