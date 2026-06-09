import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.main import app

runner = CliRunner()


def _login_result():
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def test_get_prints_transcript_text(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42"])
    assert result.exit_code == 0
    assert "retrieved text" in result.output


def test_get_output_text_prints_raw(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42", "-o", "text"])
    assert result.exit_code == 0
    assert result.output.strip() == "retrieved text"


def test_get_output_id_prints_id(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42", "-o", "id"])
    assert result.exit_code == 0
    assert result.output.strip() == "t_42"


def test_get_json_emits_full_payload(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    fake.json_response = None  # falls back to the compact summary
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["id"] == "t_42"
    assert data["text"] == "retrieved text"


def test_get_output_invalid_field_exits_2():
    config.set_api_key("default", "sk_live")
    result = runner.invoke(app, ["transcripts", "get", "t_42", "-o", "bogus"])
    assert result.exit_code == 2


def test_list_renders_rows(mocker):
    config.set_api_key("default", "sk_live")
    rows = [{"id": "t1", "status": "completed"}, {"id": "t2", "status": "processing"}]
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=rows
    )
    result = runner.invoke(app, ["transcripts", "list", "--json"])
    assert result.exit_code == 0
    assert "t1" in result.output and "t2" in result.output


def test_list_unauthenticated_runs_login(monkeypatch, mocker):
    monkeypatch.setattr("aai_cli.context.run_login_flow", _login_result)
    rows = [{"id": "t1", "status": "completed"}]
    list_ = mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=rows
    )
    result = runner.invoke(app, ["transcripts", "list", "--json"])
    assert result.exit_code == 4
    assert config.get_api_key("default") == "sk_from_oauth"
    list_.assert_not_called()
    assert "Run the same command again" in result.output


def test_list_human_mode_renders_table(monkeypatch, mocker):
    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    rows = [{"id": "t1", "status": "completed", "created": "2026-01-01"}]
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=rows
    )
    result = runner.invoke(app, ["transcripts", "list"])
    assert result.exit_code == 0
    assert "t1" in result.output  # rendered through the Rich table path


def test_get_errored_transcript_exits_nonzero(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_err"
    fake.status = "error"
    fake.error = "decode failed"
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_err"])
    assert result.exit_code == 1


def test_list_table_colors_status(monkeypatch, mocker):
    from aai_cli.theme import make_console

    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    # Force a real color terminal so styling produces ANSI we can assert on.
    monkeypatch.setattr(
        "aai_cli.output.console",
        make_console(force_terminal=True, color_system="truecolor"),
    )
    rows = [
        {"id": "t1", "status": "completed", "created": "2026-01-01"},
        {"id": "t2", "status": "error", "created": "2026-01-02"},
    ]
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=rows
    )
    result = runner.invoke(app, ["transcripts", "list"], color=True)
    assert result.exit_code == 0
    assert "completed" in result.output
    assert "error" in result.output
    assert "\x1b[1;32m" in result.output  # aai.success (bold green) → "completed" cell
    assert "\x1b[1;31m" in result.output  # aai.error (bold red) → "error" cell
