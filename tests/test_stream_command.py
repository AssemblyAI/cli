import json
import types

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def _drive_turns(api_key, source, *, sample_rate, on_begin=None, on_turn=None, on_termination=None):
    # Simulate the streaming client driving the renderer callbacks.
    if on_begin:
        on_begin(types.SimpleNamespace(id="sess"))
    if on_turn:
        on_turn(types.SimpleNamespace(transcript="hello world", end_of_turn=True))


def test_stream_help_lists_command():
    result = runner.invoke(app, ["stream", "--help"])
    assert result.exit_code == 0
    assert "microphone" in result.output.lower()


def test_stream_mic_renders_turns(monkeypatch):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", _drive_turns)
    result = runner.invoke(app, ["stream", "--json"])
    assert result.exit_code == 0
    lines = [json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "hello world", "end_of_turn": True} in lines


def test_stream_file_uses_filesource(monkeypatch, tmp_path):
    config.set_api_key("default", "sk_live")
    seen = {}

    def fake_stream_audio(
        api_key, source, *, sample_rate, on_begin=None, on_turn=None, on_termination=None
    ):
        seen["source_type"] = type(source).__name__
        seen["rate"] = sample_rate

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", fake_stream_audio)
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p)])
    assert result.exit_code == 0
    assert seen["source_type"] == "FileSource"
    assert seen["rate"] == 16000


def test_stream_unauthenticated_exits_2():
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 2


def test_stream_ctrl_c_exits_cleanly(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_kbd(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", raise_kbd)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_file_with_sample_rate_flag_rejected(tmp_path):
    config.set_api_key("default", "sk_live")
    import wave

    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p), "--sample-rate", "44100"])
    assert result.exit_code == 2


def test_stream_broken_pipe_exits_zero(monkeypatch):
    config.set_api_key("default", "sk_live")

    def raise_broken_pipe(*a, **k):
        raise BrokenPipeError

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", raise_broken_pipe)
    result = runner.invoke(app, ["stream"])
    assert result.exit_code == 0


def test_stream_file_json_output(monkeypatch, tmp_path):
    import json as _json
    import wave

    config.set_api_key("default", "sk_live")

    def fake(api_key, source, *, sample_rate, on_begin=None, on_turn=None, on_termination=None):
        if on_turn:
            on_turn(types.SimpleNamespace(transcript="from file", end_of_turn=True))

    monkeypatch.setattr("assemblyai_cli.commands.stream.client.stream_audio", fake)
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x01" * 100)
    result = runner.invoke(app, ["stream", str(p), "--json"])
    assert result.exit_code == 0
    lines = [_json.loads(x) for x in result.output.splitlines() if x.strip()]
    assert {"type": "turn", "transcript": "from file", "end_of_turn": True} in lines
