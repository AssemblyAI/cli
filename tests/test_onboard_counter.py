from __future__ import annotations

from typing import ClassVar

import pytest
from typer.testing import CliRunner

from aai_cli import client, config
from aai_cli.main import app


class _FakeTranscript:
    id = "t_123"
    status = "completed"
    text = "hello world"
    json_response: ClassVar[dict[str, str]] = {"id": "t_123", "text": "hello world"}
    utterances = None


def test_transcribe_increments_request_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk_test")
    monkeypatch.setattr(client, "transcribe", lambda *a, **k: _FakeTranscript())
    # `-o text` keeps us off the rich render path (which needs a full transcript object);
    # the counter increments before the output branch either way.
    result = CliRunner().invoke(app, ["transcribe", "--sample", "-o", "text"])
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output
    assert config.get_requests_made("default") == 1
