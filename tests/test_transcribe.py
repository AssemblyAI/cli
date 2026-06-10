"""`aai transcribe` behavior: output rendering, LLM transforms, sources, --show-code.

Flag-to-config mapping and flag validation live in test_transcribe_flags.py.
"""

import json

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def audio_file(tmp_path, monkeypatch):
    # The command checks the local path exists before resolving credentials, so the
    # "audio.mp3" the tests pass must be a real file; run each test in its own cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audio.mp3").write_bytes(b"fake-audio")


def _auth():
    config.set_api_key("default", "sk_live")


def _login_result():
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def _fake_transcript(mocker):
    t = mocker.MagicMock()
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


def test_transcribe_sample_prints_text(mocker):
    _auth()
    tx = mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 0
    assert "hello world" in result.output
    audio_arg = tx.call_args.args[1]
    assert audio_arg.endswith("wildfires.mp3")


def test_transcribe_json_output(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert '"id": "t_1"' in result.output


def test_transcribe_unauthenticated_runs_login_then_transcribes(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    tx = mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 4
    assert config.get_api_key("default") == "sk_from_oauth"
    tx.assert_not_called()
    assert "Run the same command again" in result.output


def test_transcribe_output_text_field(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "text"])
    assert result.exit_code == 0
    assert result.output.strip() == "hello world"  # raw text, pipe-friendly


def test_transcribe_output_id_field(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--output", "id"])
    assert result.exit_code == 0
    assert result.output.strip() == "t_1"


def test_transcribe_output_srt_field(mocker):
    _auth()
    t = _fake_transcript(mocker)
    t.export_subtitles_srt.return_value = "1\n00:00:00,000 --> 00:00:02,000\nhello world\n"
    mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True, return_value=t)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "srt"])
    assert result.exit_code == 0
    assert "00:00:00,000 --> 00:00:02,000" in result.output  # SRT body, pipe-friendly
    t.export_subtitles_srt.assert_called_once()


def test_transcribe_output_invalid_exits_2(mocker):
    _auth()
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "-o", "bogus"])
    assert result.exit_code == 2  # unknown field rejected


def test_transcribe_reads_audio_from_stdin(monkeypatch, mocker):
    import pathlib

    _auth()
    seen = {}

    def fake_transcribe(api_key, audio, *, config):
        # The piped bytes are buffered to a temp file the SDK can upload.
        seen["bytes"] = pathlib.Path(audio).read_bytes()
        return _fake_transcript(mocker)

    monkeypatch.setattr("aai_cli.commands.transcribe.client.transcribe", fake_transcribe)
    result = runner.invoke(app, ["transcribe", "-", "-o", "text"], input=b"RIFFfake-wav-bytes")
    assert result.exit_code == 0
    assert result.output.strip() == "hello world"
    assert seen["bytes"] == b"RIFFfake-wav-bytes"


def test_transcribe_status_renders_enum_value(mocker):
    import assemblyai as aai

    _auth()
    t = _fake_transcript(mocker)
    t.status = aai.TranscriptStatus.completed
    t.json_response = None
    mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True, return_value=t)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert result.exit_code == 0
    assert '"status": "completed"' in result.output


def test_transcribe_prompt_transforms_json(monkeypatch, mocker):
    _auth()
    seen = {}

    def fake_transform(api_key, *, prompt, model, transcript_id, max_tokens, transcript_text=None):
        seen["prompt"] = prompt
        seen["model"] = model
        seen["transcript_id"] = transcript_id
        return "a short summary"

    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    monkeypatch.setattr("aai_cli.commands.transcribe.llm.transform_transcript", fake_transform)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--llm", "summarize", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["text"] == "hello world"  # raw transcript still present in JSON
    steps = data["transform"]["steps"]
    assert steps == [{"prompt": "summarize", "output": "a short summary"}]
    # The transform is injected server-side via the transcript id.
    assert seen["transcript_id"] == "t_1"
    assert seen["model"] == "claude-haiku-4-5-20251001"


def test_transcribe_chains_multiple_gateway_prompts(monkeypatch, mocker):
    _auth()
    calls = []

    def fake_transform(
        api_key, *, prompt, model, max_tokens, transcript_id=None, transcript_text=None
    ):
        calls.append(
            {"prompt": prompt, "transcript_id": transcript_id, "transcript_text": transcript_text}
        )
        return f"out({prompt})"

    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    monkeypatch.setattr("aai_cli.commands.transcribe.llm.transform_transcript", fake_transform)
    result = runner.invoke(
        app,
        [
            "transcribe",
            "audio.mp3",
            "--json",
            "--llm",
            "summarize",
            "--llm",
            "translate",
        ],
    )
    assert result.exit_code == 0
    # Step 1 runs over the transcript; step 2 chains over step 1's output.
    assert calls[0]["transcript_id"] == "t_1" and calls[0]["transcript_text"] is None
    assert calls[1]["transcript_id"] is None and calls[1]["transcript_text"] == "out(summarize)"
    steps = json.loads(result.output)["transform"]["steps"]
    assert steps == [
        {"prompt": "summarize", "output": "out(summarize)"},
        {"prompt": "translate", "output": "out(translate)"},
    ]


def test_transcribe_prompt_human_shows_only_transform(monkeypatch, mocker):
    _auth()
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    monkeypatch.setattr(
        "aai_cli.commands.transcribe.llm.transform_transcript",
        lambda *a, **k: "TRANSFORMED",
    )
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--llm", "summarize"])
    assert result.exit_code == 0
    assert "TRANSFORMED" in result.output
    assert "hello world" not in result.output  # human mode shows the transform only


def test_transcribe_chained_prompts_human_labels_each_step(monkeypatch, mocker):
    # Human render of a multi-step chain labels each step (the single-step path
    # prints only the lone output; this one enumerates "Step N").
    _auth()
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    monkeypatch.setattr(
        "aai_cli.commands.transcribe.llm.transform_transcript",
        lambda *a, **k: f"out({k['prompt']})",
    )
    result = runner.invoke(
        app,
        ["transcribe", "audio.mp3", "--llm", "summarize", "--llm", "translate"],
    )
    assert result.exit_code == 0
    assert "Step 1 — summarize:" in result.output
    assert "Step 2 — translate:" in result.output
    assert "out(summarize)" in result.output


def test_transcribe_youtube_url_downloads_then_transcribes(monkeypatch, mocker, tmp_path):
    _auth()
    fake = tmp_path / "vid.m4a"
    fake.write_bytes(b"x")
    monkeypatch.setattr("aai_cli.transcribe_exec.youtube.download_audio", lambda url, d: fake)
    tx = mocker.patch(
        "aai_cli.commands.transcribe.client.transcribe",
        autospec=True,
        return_value=_fake_transcript(mocker),
    )
    result = runner.invoke(app, ["transcribe", "https://youtu.be/abc", "--json"])
    assert result.exit_code == 0
    assert tx.call_args.args[1] == str(fake)  # transcribed the downloaded local file


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


def test_transcribe_renders_summary_human(monkeypatch, mocker):
    _auth()
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    t = _fake_transcript(mocker)
    t.summary = "three bullet summary"
    t.chapters = []
    mocker.patch("aai_cli.commands.transcribe.client.transcribe", autospec=True, return_value=t)
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--summarization"])
    assert result.exit_code == 0
    assert "Summary:" in result.output
    assert "three bullet summary" in result.output
