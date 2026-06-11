"""`assembly transcribe --show-code` print-only behavior.

Split from test_transcribe.py (output rendering, LLM transforms, sources);
flag-to-config mapping and flag validation live in test_transcribe_flags.py.
"""

from typer.testing import CliRunner

from aai_cli.main import app

runner = CliRunner()


def test_transcribe_show_code_prints_without_transcribing(monkeypatch):
    # Print-only: emits code, never calls the API, needs no auth.
    called = []
    monkeypatch.setattr(
        "aai_cli.commands.transcribe.client.transcribe",
        lambda *a, **k: called.append(True),
    )
    result = runner.invoke(app, ["transcribe", "--sample", "--speaker-labels", "--show-code"])
    assert result.exit_code == 0
    assert called == []  # never transcribed
    assert "import assemblyai as aai" in result.output
    assert "TranscriptionConfig(" in result.output
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in result.output
    # --sample resolves to the hosted sample URL (not the no-source placeholder).
    assert "wildfires" in result.output
    assert "your-audio-file.mp3" not in result.output


def test_transcribe_show_code_includes_download_sections(monkeypatch):
    # --show-code reflects --download-sections in the generated yt-dlp download block.
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", _boom)
    result = runner.invoke(
        app,
        ["transcribe", "https://youtu.be/abc", "--download-sections", "*0:00-5:00", "--show-code"],
    )
    assert result.exit_code == 0
    assert "download_range_func([], [(0.0, 300.0)], False)" in result.output
    assert '"force_keyframes_at_cuts": True,' in result.output


def test_transcribe_show_code_without_source_uses_placeholder(monkeypatch):
    # --show-code never transcribes, so it must not demand a source/--sample; it emits
    # runnable code with a clearly-marked placeholder path instead of erroring.
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", _boom)
    result = runner.invoke(app, ["transcribe", "--show-code"])
    assert result.exit_code == 0
    assert "import assemblyai as aai" in result.output
    assert "your-audio-file.mp3" in result.output


def test_transcribe_show_code_ignores_json_flag(monkeypatch):
    # --show-code is print-only; --json does not suppress or wrap it.
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr(
        "aai_cli.commands.transcribe.client.transcribe",
        _boom,
    )
    result = runner.invoke(app, ["transcribe", "--sample", "--show-code", "--json"])
    assert result.exit_code == 0
    assert "import assemblyai as aai" in result.output


def test_transcribe_show_code_includes_llm_gateway_without_running(monkeypatch):
    # --llm must be reflected in the generated code, still without
    # transcribing or calling the gateway.
    def _boom(*a, **k):
        raise AssertionError("must not call the API")

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", _boom)
    monkeypatch.setattr("aai_cli.commands.transcribe.llm.transform_transcript", _boom)
    result = runner.invoke(
        app,
        ["transcribe", "--sample", "--llm", "translate to spanish", "--show-code"],
    )
    assert result.exit_code == 0
    assert "llm-gateway.assemblyai.com" in result.output
    assert "translate to spanish" in result.output
    assert '"transcript_id": transcript.id' in result.output


def test_transcribe_show_code_output_srt_generates_export(monkeypatch):
    # -o srt must be reflected in the generated code (not silently dropped).
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", _boom)
    result = runner.invoke(app, ["transcribe", "--sample", "-o", "srt", "--show-code"])
    assert result.exit_code == 0
    compile(result.output, "<generated>", "exec")  # the emitted script is runnable
    assert "print(transcript.export_subtitles_srt())" in result.output
    assert "print(transcript.text)" not in result.output


def test_transcribe_show_code_output_utterances_generates_loop(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not transcribe")

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", _boom)
    result = runner.invoke(app, ["transcribe", "--sample", "-o", "utterances", "--show-code"])
    assert result.exit_code == 0
    compile(result.output, "<generated>", "exec")
    assert 'print(f"Speaker {utt.speaker}: {utt.text}")' in result.output
