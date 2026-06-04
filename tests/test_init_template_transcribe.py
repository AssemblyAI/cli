import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/transcribe")


def _load_app(monkeypatch):
    """Import the template's api/index.py as a module and return its FastAPI app.

    The template is a standalone project (not part of aai_cli's import graph), so we
    load it by file path. assemblyai is stubbed so no network/key is needed.
    """
    fake_aai = MagicMock()
    fake_aai.TranscriptStatus.completed = "completed"
    fake_aai.TranscriptStatus.error = "error"
    submitted = MagicMock(id="t-123")
    fake_aai.Transcriber.return_value.submit.return_value = submitted
    done = MagicMock(status="completed", error=None,
                     json_response={"text": "hello world", "utterances": []})
    fake_aai.Transcript.get_by_id.return_value = done
    monkeypatch.setitem(sys.modules, "assemblyai", fake_aai)

    spec = importlib.util.spec_from_file_location(
        "_tmpl_transcribe", TEMPLATE_DIR / "api" / "index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app, fake_aai


def test_required_files_exist():
    for rel in ("api/index.py", "index.html", "vercel.json",
                "requirements.txt", "README.md", "gitignore", "env.example"):
        assert (TEMPLATE_DIR / rel).exists(), rel


def test_template_ships_no_real_key():
    # No committed file may contain a literal .env with a key; only env.example.
    assert not (TEMPLATE_DIR / ".env").exists()
    assert "your_assemblyai_api_key_here" in (TEMPLATE_DIR / "env.example").read_text()


def test_index_route_serves_page(monkeypatch):
    app, _ = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_submit_returns_transcript_id(monkeypatch):
    app, fake = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.post("/api/transcribe", files={"file": ("a.mp3", b"\x00\x01", "audio/mpeg")})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    fake.Transcriber.return_value.submit.assert_called_once()


def test_status_returns_completed_transcript(monkeypatch):
    app, _ = _load_app(monkeypatch)
    client = TestClient(app)
    resp = client.get("/api/status/t-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["transcript"]["text"] == "hello world"
