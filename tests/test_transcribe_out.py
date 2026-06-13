import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def audio_file(tmp_path, monkeypatch):
    # The command checks the local path exists before resolving credentials, so the
    # "audio.mp3" the tests pass must be a real file; run each test in its own cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")


_TRANSCRIBE = "aai_cli.app.transcribe_exec.client.transcribe"


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript():
    t = MagicMock()
    t.id = "t_1"
    t.text = "hello world"
    t.status = "completed"
    t.json_response = {"id": "t_1", "text": "hello world", "status": "completed"}
    for attr in (
        "summary",
        "chapters",
        "auto_highlights",
        "sentiment_analysis",
        "entities",
        "iab_categories",
        "content_safety",
    ):
        setattr(t, attr, None)
    t.utterances = None
    return t


def test_transcribe_out_writes_text_file(tmp_path):
    _auth()
    out = tmp_path / "episode.txt"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_text() == "hello world\n"
    # The transcript went to the file, not the terminal — stdout stays clean.
    assert "hello world" not in result.output
    # A confirmation is shown on stderr so the user knows where it landed.
    assert "Saved to" in result.output


def test_transcribe_out_quiet_suppresses_confirmation(tmp_path):
    # -q silences the "Saved to" confirmation, but the file is still written.
    _auth()
    out = tmp_path / "episode.txt"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(app, ["-q", "transcribe", "audio.mp3", "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_text() == "hello world\n"
    assert "Saved to" not in result.output


def test_transcribe_out_with_output_field_writes_that_field(tmp_path):
    _auth()
    out = tmp_path / "id.txt"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "id", "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_text() == "t_1\n"


def test_transcribe_out_with_json_writes_json_file(tmp_path):
    _auth()
    out = tmp_path / "t.json"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--json", "--out", str(out)])
    assert result.exit_code == 0
    assert json.loads(out.read_text())["id"] == "t_1"


def test_transcribe_out_with_llm_is_a_usage_error(tmp_path):
    # --out captures the transcript; chaining an LLM transform into a file isn't
    # supported (pipe it instead), so the combination is rejected up front.
    _auth()
    out = tmp_path / "x.txt"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(
            app, ["transcribe", "audio.mp3", "--llm", "summarize", "--out", str(out)]
        )
    assert result.exit_code == 2
    assert "--out and --llm can't be combined." in result.output
    assert "Pipe the transform instead" in result.output
    assert not out.exists()


def test_transcribe_out_missing_parent_dir_fails_before_transcribing(tmp_path):
    # The --out path is validated up front: a bad directory must not surface as an
    # internal error after a billed, possibly long transcription.
    _auth()
    out = tmp_path / "no" / "such" / "dir" / "x.txt"
    with patch(_TRANSCRIBE) as tx:
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--out", str(out)])
    assert result.exit_code == 2
    assert "doesn't exist" in result.output
    tx.assert_not_called()
    assert not out.exists()


def test_transcribe_out_unwritable_parent_dir_fails_before_transcribing(tmp_path, monkeypatch):
    import os

    from aai_cli.app import transcribe_validate

    _auth()
    out = tmp_path / "x.txt"
    calls = []
    real_access = os.access

    def fake_access(path, mode, **kwargs):
        if path == out.parent:
            calls.append(mode)
            return False
        return real_access(path, mode, **kwargs)

    monkeypatch.setattr(transcribe_validate.os, "access", fake_access)
    with patch(_TRANSCRIBE) as tx:
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--out", str(out)])
    assert result.exit_code == 2
    assert "isn't writable" in result.output
    tx.assert_not_called()
    assert calls == [os.W_OK]  # the writability probe, not a read/exec check


def test_transcribe_out_rejects_path_traversal(tmp_path):
    # A --out path with a `..` segment is rejected with a clean usage error,
    # before anything is written.
    _auth()
    out = tmp_path / ".." / "evil.txt"
    with patch(_TRANSCRIBE, return_value=_fake_transcript()):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--out", str(out)])
    assert result.exit_code == 2
    assert "can't contain" in result.output
    assert not out.exists()
