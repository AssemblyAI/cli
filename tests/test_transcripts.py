from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from assemblyai_cli import config
from assemblyai_cli.main import app

runner = CliRunner()


def test_get_prints_transcript_text():
    config.set_api_key("default", "sk_live")
    fake = MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    with patch("assemblyai_cli.commands.transcripts.client.get_transcript", return_value=fake):
        result = runner.invoke(app, ["get", "t_42"])
    assert result.exit_code == 0
    assert "retrieved text" in result.output


def test_list_renders_rows():
    config.set_api_key("default", "sk_live")
    rows = [{"id": "t1", "status": "completed"}, {"id": "t2", "status": "processing"}]
    with patch("assemblyai_cli.commands.transcripts.client.list_transcripts", return_value=rows):
        result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0
    assert "t1" in result.output and "t2" in result.output


def test_list_unauthenticated_exits_2():
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 2


def test_get_errored_transcript_exits_nonzero():
    config.set_api_key("default", "sk_live")
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.id = "t_err"
    fake.status = "error"
    fake.error = "decode failed"
    with patch("assemblyai_cli.commands.transcripts.client.get_transcript", return_value=fake):
        result = runner.invoke(app, ["get", "t_err"])
    assert result.exit_code == 1
