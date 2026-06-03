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
    t.export_subtitles_srt.return_value = "1\n00:00\nhello"
    t.export_subtitles_vtt.return_value = "WEBVTT\nhello"
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


def test_transcribe_passes_speaker_labels():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ) as tx:
        runner.invoke(app, ["transcribe", "audio.mp3", "--speaker-labels"])
    assert tx.call_args.kwargs["speaker_labels"] is True


def test_transcribe_srt_export():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--srt"])
    assert "00:00" in result.output


def test_transcribe_vtt_export():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--vtt"])
    assert "WEBVTT" in result.output


def test_transcribe_srt_vtt_mutually_exclusive():
    _auth()
    with patch(
        "assemblyai_cli.commands.transcribe.client.transcribe", return_value=_fake_transcript()
    ):
        result = runner.invoke(app, ["transcribe", "audio.mp3", "--srt", "--vtt"])
    assert result.exit_code != 0


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


def test_transcribe_srt_vtt_conflict_json_error():
    _auth()
    result = runner.invoke(app, ["transcribe", "audio.mp3", "--srt", "--vtt", "--json"])
    assert result.exit_code == 2
    # In --json mode the error is a JSON envelope, not Typer usage text.
    assert '"error"' in result.output
