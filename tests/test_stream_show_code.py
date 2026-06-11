"""`assembly stream --show-code` print-only behavior.

Split from test_stream_command.py (source/streaming behavior); flag-to-params
mapping and conflicting-flag validation live in test_stream_command_flags.py.
"""

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_stream_show_code_prints_without_streaming(monkeypatch):
    # Print-only: emits the mic-streaming script, never opens audio or streams, no auth.
    called = []
    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        lambda *a, **k: called.append(True),
    )
    result = runner.invoke(app, ["stream", "--show-code"])
    assert result.exit_code == 0
    assert called == []  # never streamed
    assert "StreamingClient(" in result.output
    assert "MicrophoneStream(sample_rate=16000)" in result.output
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in result.output


def test_stream_show_code_file_source_streams_that_file():
    # A file source must generate file-streaming code, not silently emit mic code.
    # The file need not exist: generating code for it is legitimate (check_local=False).
    result = runner.invoke(app, ["stream", "rec.wav", "--show-code"])
    assert result.exit_code == 0
    assert "client.stream(file_chunks())" in result.output
    assert "'rec.wav'" in result.output  # the ffmpeg input is the file passed
    assert "MicrophoneStream" not in result.output
    compile(result.output, "<show-code>", "exec")  # the printed script is runnable


def test_stream_show_code_stdin_source_reads_stdin():
    result = runner.invoke(app, ["stream", "-", "--show-code"])
    assert result.exit_code == 0
    assert "client.stream(stdin_chunks())" in result.output
    assert "sys.stdin.buffer" in result.output
    assert "MicrophoneStream" not in result.output
    compile(result.output, "<show-code>", "exec")


def test_stream_show_code_sample_streams_hosted_clip():
    result = runner.invoke(app, ["stream", "--sample", "--show-code"])
    assert result.exit_code == 0
    assert "wildfires.mp3" in result.output
    assert "file_chunks()" in result.output
    assert "MicrophoneStream" not in result.output


def test_stream_show_code_honors_sample_rate_flag():
    result = runner.invoke(app, ["stream", "--sample-rate", "8000", "--show-code"])
    assert result.exit_code == 0
    assert "MicrophoneStream(sample_rate=8000)" in result.output
    assert "sample_rate=8000," in result.output  # params match the capture rate


def test_stream_show_code_honors_config_sample_rate():
    # An explicit `--config sample_rate=…` must not be overridden by the 16 kHz default.
    result = runner.invoke(app, ["stream", "--config", "sample_rate=8000", "--show-code"])
    assert result.exit_code == 0
    assert "MicrophoneStream(sample_rate=8000)" in result.output
    assert "sample_rate=8000," in result.output


def test_stream_show_code_sample_rate_flag_beats_config():
    result = runner.invoke(
        app, ["stream", "--sample-rate", "8000", "--config", "sample_rate=44100", "--show-code"]
    )
    assert result.exit_code == 0
    assert "MicrophoneStream(sample_rate=8000)" in result.output
    assert "44100" not in result.output


def test_stream_show_code_file_with_mic_flags_rejected():
    # --show-code applies the same source validation as a real run, so the
    # file + --sample-rate conflict errors instead of generating mic code.
    result = runner.invoke(app, ["stream", "rec.wav", "--sample-rate", "8000", "--show-code"])
    assert result.exit_code == 2
    assert "--sample-rate" in result.output


def test_stream_show_code_rejects_downloaded_sources():
    # YouTube and podcast-page URLs both route through the download path, which the
    # generated streaming script doesn't model yet.
    for url in ("https://youtu.be/abc", "https://www.spreaker.com/episode/12345"):
        result = runner.invoke(app, ["stream", url, "--show-code"])
        assert result.exit_code == 2
        assert "downloaded sources" in result.output
        assert "YouTube" in result.output


def test_stream_show_code_ignores_json_flag(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not stream")

    monkeypatch.setattr(
        "aai_cli.commands.stream.client.stream_audio",
        _boom,
    )
    result = runner.invoke(app, ["stream", "--show-code", "--json"])
    assert result.exit_code == 0
    assert "StreamingClient(" in result.output
