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
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
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


def test_list_limit_must_be_at_least_one(mocker):
    # min=1 on --limit: 0 and negatives are rejected client-side, before any request.
    config.set_api_key("default", "sk_live")
    list_ = mocker.patch("aai_cli.commands.transcripts.client.list_transcripts", autospec=True)
    for bad in ("0", "-3"):
        result = runner.invoke(app, ["transcripts", "list", "--limit", bad])
        assert result.exit_code == 2
        assert "limit" in result.output.lower()
    list_.assert_not_called()


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
    # The transcript's own error message is surfaced, not the generic fallback
    # (pins `getattr(transcript, "error", None) or "Transcript failed."`).
    assert "decode failed" in result.output


def test_list_table_colors_status(monkeypatch, mocker):
    from aai_cli.theme import make_console

    config.set_api_key("default", "sk_live")
    monkeypatch.setattr("aai_cli.output.resolve_json", lambda *, explicit: False)
    # Pin a truecolor console with an empty _environ so the rendered ANSI is
    # deterministic: Rich otherwise reads ambient color env (NO_COLOR/COLORTERM/...)
    # at render time, which leaks across tests and flips the color depth. With
    # _environ={} the depth is fixed by color_system alone.
    monkeypatch.setattr(
        "aai_cli.output.console",
        make_console(force_terminal=True, color_system="truecolor", _environ={}),
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
    assert "\x1b[1;38;2;240;68;56m" in result.output  # aai.error (bold #F04438) → "error" cell
