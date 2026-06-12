"""`assembly stream` flag handling: flag-to-params mapping and conflicting-flag validation.

Source/streaming behavior and --show-code tests live in test_stream_command.py.
"""

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.main import app

runner = CliRunner()


def test_stream_maps_turn_detection_flags(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}

    def fake_stream_audio(api_key, source, *, params, **kw):
        captured["params"] = params

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", fake_stream_audio)

    runner.invoke(
        app,
        [
            "stream",
            "--sample",
            "--max-turn-silence",
            "400",
            "--filter-profanity",
            "--speaker-labels",
        ],
    )
    params = captured["params"]
    assert params.max_turn_silence == 400
    assert params.filter_profanity is True
    assert params.speaker_labels is True


def test_stream_config_escape_hatch(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(app, ["stream", "--sample", "--config", "vad_threshold=0.7"])
    assert captured["params"].vad_threshold == 0.7


def test_stream_maps_webhook_auth_header(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(
        app,
        [
            "stream",
            "--sample",
            "--webhook-url",
            "https://example.com/hook",
            "--webhook-auth-header",
            "Authorization:Bearer xyz",
        ],
    )
    params = captured["params"]
    assert params.webhook_auth_header_name == "Authorization"
    assert params.webhook_auth_header_value == "Bearer xyz"


def test_stream_format_turns_tristate(monkeypatch):
    config.set_api_key("default", "sk_live")
    captured = {}
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda api_key, source, *, params, **kw: captured.update(params=params),
    )

    runner.invoke(app, ["stream", "--sample"])
    assert captured["params"].format_turns is True  # unset defaults to True

    runner.invoke(app, ["stream", "--sample", "--no-format-turns"])
    assert captured["params"].format_turns is False


def test_stream_sample_with_sample_rate_rejected():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "--sample", "--sample-rate", "44100"])
    assert result.exit_code == 2  # mic-only flags don't apply to a file/sample source


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


def test_stream_json_with_text_output_is_usage_error():
    # Contradictory output shapes (--json + -o text) are rejected up front, before
    # credentials, like the --llm + -o text precedent.
    result = runner.invoke(app, ["stream", "--json", "-o", "text"])
    assert result.exit_code == 2
    assert "--json and -o text can't be combined." in result.output
    assert "Pick one output format." in result.output


def test_stream_stdin_with_sample_rejected():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "-", "--sample"], input=b"\x00\x00")
    assert result.exit_code == 2
    assert "--sample" in result.output


def test_stream_file_source_with_sample_rejected(monkeypatch, tmp_path):
    # A real source plus --sample is a conflict (the file would silently lose),
    # surfaced by resolve_audio_source as a usage error before any streaming.
    config.set_api_key("default", "sk_live")

    def _boom(*a, **k):
        raise AssertionError("must not stream a conflicting source")

    monkeypatch.setattr("aai_cli.commands.stream.client.stream_audio", _boom)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    result = runner.invoke(app, ["stream", str(wav), "--sample"])
    assert result.exit_code == 2
    assert "--sample" in result.output
    assert "cannot be combined" in result.output


def test_stream_stdin_rejects_device(monkeypatch):
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["stream", "-", "--device", "2"], input=b"\x00\x00")
    assert result.exit_code == 2  # --device applies only to the microphone
