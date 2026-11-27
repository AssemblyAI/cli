"""BOM and ordinary UTF-8 CSVs have identical column and scoring semantics."""

from __future__ import annotations

import wave
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from aai_cli.commands.evaluate import _data as eval_data
from aai_cli.core import client, config
from aai_cli.main import app


@pytest.mark.parametrize("bom", [b"", b"\xef\xbb\xbf"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_csv_encoding_preserves_columns_and_wer(tmp_path, monkeypatch, bom, explicit, newline):
    audio = tmp_path / "call.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    manifest = tmp_path / "calls.csv"
    manifest.write_bytes(
        bom + f'"audio","text"{newline}"call.wav","héllo, world"{newline}'.encode()
    )
    data = eval_data.load(
        str(manifest),
        audio_column="audio" if explicit else None,
        text_column="text" if explicit else None,
        limit=10,
    )
    assert data.items == [eval_data.EvalItem("call.wav", str(audio), "héllo, world")]
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")
    calls = []

    def transcribe(api_key, source, *, config):
        calls.append(source)
        return SimpleNamespace(text="héllo world")

    monkeypatch.setattr(client, "transcribe", transcribe)
    args = ["eval", str(manifest), "--json"]
    if explicit:
        args += ["--audio-column", "audio", "--text-column", "text"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [str(audio)]
    assert '"words": 2' in result.output
    assert '"errors": 0' in result.output
    assert '"wer": 0.0' in result.output
    assert "\ufeff" not in result.output
