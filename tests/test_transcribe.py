import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript():
    t = MagicMock()
    t.id = "t_1"
    t.text = "hello world"
    t.status = "completed"
    t.utterances = []
    return t


def test_transcribe_sample_prints_text():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 0
    assert "hello world" in result.output
    audio_arg = tx.call_args.args[1]
    assert audio_arg.endswith("wildfires.mp3")


def test_transcribe_requires_source():
    _auth()
    result = runner.invoke(app, ["transcribe"])
    assert result.exit_code == 2


def test_transcribe_speaker_labels_renders_utterances(monkeypatch):
    import types

    _auth()
    t = _fake_transcript()
    t.utterances = [
        types.SimpleNamespace(speaker="A", text="hi there", start=0, end=500),
        types.SimpleNamespace(speaker="B", text="hello back", start=500, end=900),
    ]
    # human mode -> "Speaker A: ..." lines
    monkeypatch.setattr("assemblyai_cli.output.resolve_json", lambda *, explicit: False)
    with patch("assemblyai_cli.commands.transcribe.client.transcribe", return_value=t):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert result.exit_code == 0
    assert "Speaker A: hi there" in result.output
    assert "Speaker B: hello back" in result.output


def test_transcribe_speaker_labels_json_includes_utterances():
    import types

    _auth()
    t = _fake_transcript()
    t.utterances = [types.SimpleNamespace(speaker="A", text="hi there", start=0, end=500)]
    with patch("assemblyai_cli.commands.transcribe.client.transcribe", return_value=t):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["utterances"] == [{"speaker": "A", "text": "hi there", "start": 0, "end": 500}]


def test_transcribe_passes_speaker_labels():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert tx.call_args.kwargs["speaker_labels"] is True


def test_transcribe_json_output():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert '"id": "t_1"' in result.output


def test_transcribe_unauthenticated_exits_2():
    result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 2


def test_transcribe_status_renders_enum_value():
    import assemblyai as aai

    _auth()
    t = _fake_transcript()
    t.status = aai.TranscriptStatus.completed
    with patch("assemblyai_cli.commands.transcribe.client.transcribe", return_value=t):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--json"])
    assert result.exit_code == 0
    assert '"status": "completed"' in result.output


def test_transcribe_prompt_transforms_json(monkeypatch):
    _auth()
    seen = {}

    def fake_transform(api_key, *, prompt, model, transcript_id, max_tokens):
        seen["prompt"] = prompt
        seen["model"] = model
        seen["transcript_id"] = transcript_id
        return "a short summary"

    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        monkeypatch.setattr(
            "assemblyai_cli.commands.transcribe.llm.transform_transcript", fake_transform
        )
        result = runner.invoke(
            app, ["transcribe", "audio.mp3", "--llm-gateway-prompt", "summarize", "--json"]
        )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["text"] == "hello world"  # raw transcript still present in JSON
    assert data["transform"]["output"] == "a short summary"
    assert data["transform"]["prompt"] == "summarize"
    # The transform is injected server-side via the transcript id.
    assert seen["transcript_id"] == "t_1"
    assert seen["model"] == "claude-sonnet-4-6"


def test_transcribe_prompt_human_shows_only_transform(monkeypatch):
    _auth()
    monkeypatch.setattr("assemblyai_cli.output.resolve_json", lambda *, explicit: False)
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        monkeypatch.setattr(
            "assemblyai_cli.commands.transcribe.llm.transform_transcript",
            lambda *a, **k: "TRANSFORMED",
        )
        result = runner.invoke(
            app, ["transcribe", "audio.mp3", "--llm-gateway-prompt", "summarize"]
        )
    assert result.exit_code == 0
    assert "TRANSFORMED" in result.output
    assert "hello world" not in result.output  # human mode shows the transform only


def test_transcribe_youtube_url_downloads_then_transcribes(monkeypatch, tmp_path):
    _auth()
    fake = tmp_path / "vid.m4a"
    fake.write_bytes(b"x")
    monkeypatch.setattr(
        "assemblyai_cli.commands.transcribe.youtube.download_audio", lambda url, d: fake
    )
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "https://youtu.be/abc", "--json"])
    assert result.exit_code == 0
    assert tx.call_args.args[1] == str(fake)  # transcribed the downloaded local file


def test_transcribe_prompt_biases_speech_model():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--prompt", "expect medical terms"])
    assert result.exit_code == 0
    # --prompt is the speech-model prompt, forwarded to the transcription call.
    assert tx.call_args.kwargs["prompt"] == "expect medical terms"


def test_render_transcript_colors_speaker_labels():
    import io

    from assemblyai_cli import theme
    from assemblyai_cli.commands.transcribe import _render_transcript

    data = {
        "text": "ignored when utterances present",
        "utterances": [
            {"speaker": "A", "text": "hello", "start": 0, "end": 1},
            {"speaker": "B", "text": "hi there", "start": 1, "end": 2},
        ],
    }
    rendered = _render_transcript(data)
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, color_system="truecolor")
    console.print(rendered)
    out = buf.getvalue()
    assert "Speaker A:" in out
    assert "hello" in out
    assert "Speaker B:" in out
    assert "\x1b[" in out  # speaker labels are styled


def test_render_transcript_plain_text_unchanged():
    from assemblyai_cli.commands.transcribe import _render_transcript

    assert _render_transcript({"text": "just the words"}) == "just the words"
