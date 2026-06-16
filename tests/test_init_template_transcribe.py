import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

TEMPLATE_DIR = Path("aai_cli/init/templates/audio_transcription")


def _load_app(monkeypatch, mocker):
    """Import the template's api/index.py as a module and return its FastAPI app.

    The template is a standalone project (not part of aai_cli's import graph), so we
    load it by file path. The assemblyai SDK is stubbed so no network is needed; a dummy
    key is injected so the endpoints' ``_require_key`` guard passes and they reach the
    stubbed SDK (isolate_env strips any ambient key).
    """
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test-key")
    fake_aai = mocker.MagicMock()
    fake_aai.TranscriptStatus.completed = "completed"
    fake_aai.TranscriptStatus.error = "error"
    submitted = mocker.MagicMock(id="t-123")
    fake_aai.Transcriber.return_value.submit.return_value = submitted

    # The template fetches status via the SDK's non-blocking api.get_transcript,
    # so stub the assemblyai.api and assemblyai.client submodules too.
    fake_api = mocker.MagicMock()
    done = mocker.MagicMock(status="completed", error=None)
    done.dict.return_value = {"text": "hello world", "utterances": []}
    fake_api.get_transcript.return_value = done
    fake_client = mocker.MagicMock()

    monkeypatch.setitem(sys.modules, "assemblyai", fake_aai)
    monkeypatch.setitem(sys.modules, "assemblyai.api", fake_api)
    monkeypatch.setitem(sys.modules, "assemblyai.client", fake_client)
    for name in ("api.index", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))

    module = importlib.import_module("api.index")
    return module.app, fake_aai, fake_api


def test_required_files_exist():
    for rel in (
        "api/index.py",
        "api/settings.py",
        "static/index.html",
        "static/app.js",
        "static/styles.css",
        "requirements.txt",
        "README.md",
        "AGENTS.md",
        "gitignore",
        "env.example",
    ):
        assert (TEMPLATE_DIR / rel).exists(), rel


def test_template_ships_no_real_key():
    assert not (TEMPLATE_DIR / ".env").exists()
    assert "your_assemblyai_api_key_here" in (TEMPLATE_DIR / "env.example").read_text(
        encoding="utf-8"
    )


def test_base_url_env_is_applied(monkeypatch, mocker):
    # assembly init writes ASSEMBLYAI_BASE_URL so a sandbox key targets the sandbox host.
    monkeypatch.setenv("ASSEMBLYAI_BASE_URL", "https://api.sb.example")
    _app, fake, _api = _load_app(monkeypatch, mocker)
    assert fake.settings.base_url == "https://api.sb.example"


def test_page_explores_all_features_and_speakers():
    # Guard the UI surface: each audio-intelligence view + per-speaker coloring stay wired.
    html = (TEMPLATE_DIR / "static" / "index.html").read_text(encoding="utf-8")
    app_js = (TEMPLATE_DIR / "static" / "app.js").read_text(encoding="utf-8")
    ui_src = html + app_js
    for token in (
        "chapters",
        "sentiment_analysis_results",
        "entities",
        "auto_highlights_result",
        "speakerColor",
        "/api/ask",  # the LLM Gateway Q&A panel
    ):
        assert token in ui_src, token


def test_ask_calls_llm_gateway_with_transcript_id(monkeypatch, mocker):
    # Stub the OpenAI SDK before the template module imports it.
    fake_openai = mocker.MagicMock()
    msg = mocker.MagicMock()
    msg.content = "It's about wildfires."
    fake_openai.OpenAI.return_value.chat.completions.create.return_value = mocker.MagicMock(
        choices=[mocker.MagicMock(message=msg)]
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    app, _aai, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.post("/api/ask", json={"transcript_id": "t-1", "question": "What is this about?"})
    assert resp.status_code == 200
    assert "wildfires" in resp.json()["answer"]
    # The gateway injects the transcript server-side via extra_body, not inline.
    kwargs = fake_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"] == {"transcript_id": "t-1"}


def test_index_route_serves_page(monkeypatch, mocker):
    app, _aai, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_submit_returns_transcript_id(monkeypatch, mocker):
    app, fake, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.post("/api/transcribe", files={"file": ("a.mp3", b"\x00\x01", "audio/mpeg")})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    fake.Transcriber.return_value.submit.assert_called_once()


def test_transcribe_url_submits_given_url(monkeypatch, mocker):
    app, fake, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
    assert resp.status_code == 200
    assert resp.json() == {"id": "t-123"}
    submitted = fake.Transcriber.return_value.submit.call_args[0][0]
    assert submitted == "https://example.com/a.mp3"  # URL passed straight through, no upload


def test_transcribe_url_defaults_to_sample(monkeypatch, mocker):
    app, fake, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={})  # no url -> the bundled sample
    assert resp.status_code == 200
    submitted = fake.Transcriber.return_value.submit.call_args[0][0]
    assert "wildfires" in submitted


def test_status_returns_completed_transcript(monkeypatch, mocker):
    app, _aai, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.get("/api/status/t-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["transcript"]["text"] == "hello world"


def test_status_returns_processing_when_not_done(monkeypatch, mocker):
    # A poll endpoint MUST report a non-terminal status without blocking. This guards
    # against regressing to the blocking aai.Transcript.get_by_id() (which would only
    # ever return completed/error, making this branch dead code).
    app, _aai, fake_api = _load_app(monkeypatch, mocker)
    fake_api.get_transcript.return_value = mocker.MagicMock(status="processing", error=None)
    client = TestClient(app)
    resp = client.get("/api/status/t-xyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "processing"}


def test_status_returns_502_on_error(monkeypatch, mocker):
    app, _aai, fake_api = _load_app(monkeypatch, mocker)
    fake_api.get_transcript.return_value = mocker.MagicMock(status="error", error="bad audio")
    client = TestClient(app)
    resp = client.get("/api/status/t-bad")
    assert resp.status_code == 502


def test_submit_surfaces_sdk_exception_as_502_not_500(monkeypatch, mocker):
    # A missing/invalid key makes the SDK raise; the endpoint must return a clean 502,
    # not let the exception bubble into a 500 traceback.
    app, fake, _api = _load_app(monkeypatch, mocker)
    fake.Transcriber.return_value.submit.side_effect = ValueError("Please provide an API key")
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
    assert resp.status_code == 502


def test_status_surfaces_sdk_exception_as_502_not_500(monkeypatch, mocker):
    app, _aai, fake_api = _load_app(monkeypatch, mocker)
    fake_api.get_transcript.side_effect = RuntimeError("network down")
    client = TestClient(app)
    resp = client.get("/api/status/t-x")
    assert resp.status_code == 502


def test_submit_returns_502_when_no_id(monkeypatch, mocker):
    # The SDK returned a transcript object without an id — surface a clean 502.
    app, fake, _api = _load_app(monkeypatch, mocker)
    fake.Transcriber.return_value.submit.return_value = mocker.MagicMock(id=None)
    client = TestClient(app)
    resp = client.post("/api/transcribe-url", json={"url": "https://example.com/a.mp3"})
    assert resp.status_code == 502
    assert "did not return an id" in resp.json()["detail"]


def test_ask_surfaces_gateway_error_as_502(monkeypatch, mocker):
    # A gateway failure (e.g. no plan access) becomes a clean 502, not a 500 traceback.
    fake_openai = mocker.MagicMock()
    fake_openai.OpenAI.return_value.chat.completions.create.side_effect = RuntimeError(
        "no plan access"
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    app, _aai, _api = _load_app(monkeypatch, mocker)
    client = TestClient(app)
    resp = client.post("/api/ask", json={"transcript_id": "t-1", "question": "what?"})
    assert resp.status_code == 502
    assert "LLM Gateway error" in resp.json()["detail"]
