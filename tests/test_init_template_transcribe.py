import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/transcribe")


def _load_app(monkeypatch):
    """Import the template's api/index.py as a module and return its FastAPI app.

    The template is a standalone project (not part of aai_cli's import graph), so we
    load it by file path. The assemblyai SDK is stubbed so no network/key is needed.
    """
    fake_aai = MagicMock()
    fake_aai.TranscriptStatus.completed = "completed"
    fake_aai.TranscriptStatus.error = "error"
    submitted = MagicMock(id="t-123")
    fake_aai.Transcriber.return_value.submit.return_value = submitted

    # The template fetches status via the SDK's non-blocking api.get_transcript,
    # so stub the assemblyai.api and assemblyai.client submodules too.
    fake_api = MagicMock()
    done = MagicMock(status="completed", error=None)
    done.dict.return_value = {"text": "hello world", "utterances": []}
    fake_api.get_transcript.return_value = done
    fake_client = MagicMock()

    monkeypatch.setitem(sys.modules, "assemblyai", fake_aai)
    monkeypatch.setitem(sys.modules, "assemblyai.api", fake_api)
    monkeypatch.setitem(sys.modules, "assemblyai.client", fake_client)

    spec = importlib.util.spec_from_file_location(
        "_tmpl_transcribe", TEMPLATE_DIR / "api" / "index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app, fake_aai, fake_api


def test_required_files_exist():
    for rel in (
        "api/index.py",
        "index.html",
        "vercel.json",
        "requirements.txt",
        "README.md",
        "gitignore",
        "env.example",
    ):
        assert (TEMPLATE_DIR / rel).exists(), rel


def test_template_ships_no_real_key():
    assert not (TEMPLATE_DIR / ".env").exists()
    assert "your_assemblyai_api_key_here" in (TEMPLATE_DIR / "env.example").read_text()


def test_index_route_serves_page(monkeypatch):
    app, _aai, _api = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_submit_returns_transcript_id(monkeypatch):
    app, fake, _api = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/transcribe", files={"file": ("a.mp3", b"\x00\x01", "audio/mpeg")})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    fake.Transcriber.return_value.submit.assert_called_once()


def test_transcribe_url_submits_given_url(monkeypatch):
    app, fake, _api = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    submitted = fake.Transcriber.return_value.submit.call_args[0][0]
    assert submitted == "https://example.com/a.mp3"  # URL passed straight through, no upload


def test_transcribe_url_defaults_to_sample(monkeypatch):
    app, fake, _api = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={})  # no url -> the bundled sample
    assert resp.status_code == 200
    submitted = fake.Transcriber.return_value.submit.call_args[0][0]
    assert "wildfires" in submitted


def test_status_returns_completed_transcript(monkeypatch):
    app, _aai, _api = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/api/status/t-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["transcript"]["text"] == "hello world"


def test_status_returns_processing_when_not_done(monkeypatch):
    # A poll endpoint MUST report a non-terminal status without blocking. This guards
    # against regressing to the blocking aai.Transcript.get_by_id() (which would only
    # ever return completed/error, making this branch dead code).
    app, _aai, fake_api = _load_app(monkeypatch)
    fake_api.get_transcript.return_value = MagicMock(status="processing", error=None)
    client = TestClient(app)
    resp = client.get("/api/status/t-xyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "processing"}


def test_status_returns_502_on_error(monkeypatch):
    app, _aai, fake_api = _load_app(monkeypatch)
    fake_api.get_transcript.return_value = MagicMock(status="error", error="bad audio")
    client = TestClient(app)
    resp = client.get("/api/status/t-bad")
    assert resp.status_code == 502
