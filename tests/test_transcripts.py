import json

from typer.testing import CliRunner

from aai_cli import config
from aai_cli.auth.flow import LoginResult
from aai_cli.main import app

runner = CliRunner()


def _login_result(*, json_mode=False):
    return LoginResult(
        api_key="sk_from_oauth", session_jwt="jwt", session_token="tok", account_id=7
    )


def test_transcripts_help_lists_list_before_get():
    # Subcommand order matches `assembly sessions --help`: list first, then get.
    result = runner.invoke(app, ["transcripts", "--help"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    list_idx = next(i for i, line in enumerate(lines) if "List recent transcripts" in line)
    get_idx = next(i for i, line in enumerate(lines) if "Fetch a past transcript" in line)
    assert list_idx < get_idx


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
    # Human mode prints the bare transcript text — not a JSON object (pins json_mode=False).
    assert result.output.strip() == "retrieved text"


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


def test_get_output_vtt_forwards_chars_per_caption(mocker):
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.status = "completed"
    fake.export_subtitles_vtt.return_value = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhi\n"
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(
        app, ["transcripts", "get", "t_42", "-o", "vtt", "--chars-per-caption", "42"]
    )
    assert result.exit_code == 0
    assert "WEBVTT" in result.output
    fake.export_subtitles_vtt.assert_called_once_with(chars_per_caption=42)


def test_get_chars_per_caption_requires_subtitle_output(mocker):
    config.set_api_key("default", "sk_live")
    get = mocker.patch("aai_cli.commands.transcripts.client.get_transcript", autospec=True)
    result = runner.invoke(app, ["transcripts", "get", "t_42", "--chars-per-caption", "42"])
    assert result.exit_code == 2
    assert "--chars-per-caption only applies to subtitle output" in result.output
    get.assert_not_called()  # rejected before any fetch


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


def test_get_json_emits_full_sdk_payload_when_present(mocker):
    # `transcripts get --json` returns the full SDK payload (same shape as
    # `transcribe --json`), so a fetched transcript round-trips for scripting.
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "retrieved text"
    fake.status = "completed"
    fake.json_response = {
        "id": "t_42",
        "status": "completed",
        "text": "retrieved text",
        "utterances": [{"speaker": "A", "text": "retrieved text"}],
    }
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["utterances"] == [{"speaker": "A", "text": "retrieved text"}]


def test_get_short_json_flag_emits_json(mocker):
    # The shared -j alias for --json works on every command.
    config.set_api_key("default", "sk_live")
    fake = mocker.MagicMock()
    fake.id = "t_42"
    fake.text = "hi"
    fake.status = "completed"
    fake.json_response = None
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript", autospec=True, return_value=fake
    )
    result = runner.invoke(app, ["transcripts", "get", "t_42", "-j"])
    assert result.exit_code == 0
    assert json.loads(result.output)["id"] == "t_42"


def test_list_empty_shows_human_empty_state(mocker):
    config.set_api_key("default", "sk_live")
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=[]
    )
    result = runner.invoke(app, ["transcripts", "list"])
    assert result.exit_code == 0
    assert "No transcripts yet." in result.output


def test_get_malformed_id_is_rejected_before_auth(monkeypatch, mocker):
    # No key configured: the cheap local id check must win over auth, so the user
    # is told to fix the id instead of being sent through login first.
    monkeypatch.setattr("aai_cli.context._interactive_session", lambda: True)
    login = mocker.patch("aai_cli.auth.run_login_flow", side_effect=AssertionError("no login"))
    get = mocker.patch("aai_cli.commands.transcripts.client.get_transcript", autospec=True)
    result = runner.invoke(app, ["transcripts", "get", "not-a-real-id!!"])
    assert result.exit_code == 2
    assert "doesn't look like a transcript id" in result.output
    get.assert_not_called()
    login.assert_not_called()


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
    monkeypatch.setattr("aai_cli.auth.run_login_flow", _login_result)
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


def test_list_human_mode_renders_table(mocker):
    config.set_api_key("default", "sk_live")
    rows = [{"id": "t1", "status": "completed", "created": "2026-01-01"}]
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts", autospec=True, return_value=rows
    )
    result = runner.invoke(app, ["transcripts", "list"])
    assert result.exit_code == 0
    assert "t1" in result.output  # rendered through the Rich table path
    assert "2026-01-01 00:00:00" in result.output  # created normalized to UTC datetime


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
