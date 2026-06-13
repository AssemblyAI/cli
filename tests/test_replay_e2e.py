"""End-to-end replay tests: drive real CLI commands against recorded API responses.

Each test patches the command's network boundary (``client.* / llm.* / ams.*``) to
return an object rebuilt from a real, scrubbed fixture (see ``tests/replay_fixtures.py``
and ``scripts/record_fixtures.py``), then invokes the command through Typer and asserts
on the rendered output. The transport stays offline — pytest-socket is untouched — but
the command's own parsing, formatting, and rendering all run against a real payload.
"""

from __future__ import annotations

from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app
from tests import replay_fixtures as rf

runner = CliRunner()


def _with_api_key():
    config.set_api_key("default", "sk_live")


def _with_session():
    config.set_session("default", session_jwt="jwt", session_token="tok", account_id=12345)


def test_transcribe_sample_renders_real_transcript(mocker):
    _with_api_key()
    mocker.patch(
        "aai_cli.app.transcribe_exec.client.transcribe",
        autospec=True,
        return_value=rf.transcript("transcribe_sample"),
    )
    result = runner.invoke(app, ["transcribe", "--sample"])
    assert result.exit_code == 0
    assert "Smoke from hundreds of wildfires" in result.output


def test_transcripts_get_renders_real_text(mocker):
    _with_api_key()
    mocker.patch(
        "aai_cli.commands.transcripts.client.get_transcript",
        autospec=True,
        return_value=rf.transcript("transcript_get"),
    )
    result = runner.invoke(app, ["transcripts", "get", "e5a56f4f-b658-44b0-925a-f7d761ec0d96"])
    assert result.exit_code == 0
    assert "Smoke from hundreds of wildfires" in result.output


def test_transcripts_list_renders_real_rows(mocker):
    _with_api_key()
    mocker.patch(
        "aai_cli.commands.transcripts.client.list_transcripts",
        autospec=True,
        return_value=rf.load_list("transcripts_list"),
    )
    result = runner.invoke(app, ["transcripts", "list"])
    assert result.exit_code == 0
    # The recorded history mixes statuses (a YouTube download failed) under the standard
    # table headers, so both the real statuses and the header render.
    assert "completed" in result.output
    assert "error" in result.output
    assert "status" in result.output


def test_llm_renders_real_completion(mocker):
    _with_api_key()
    mocker.patch(
        "aai_cli.commands.llm.gateway.complete",
        autospec=True,
        return_value=rf.completion("llm_complete"),
    )
    result = runner.invoke(app, ["llm", "Reply with exactly one word: PONG"])
    assert result.exit_code == 0
    assert "PONG" in result.output


def test_balance_renders_real_dollars(mocker):
    _with_session()
    mocker.patch(
        "aai_cli.commands.account.ams.get_balance",
        autospec=True,
        return_value=rf.load_object("account_balance"),
    )
    result = runner.invoke(app, ["balance"])
    assert result.exit_code == 0
    assert "$879.58" in result.output


def test_usage_renders_real_breakdown(mocker):
    _with_session()
    mocker.patch(
        "aai_cli.commands.account.ams.get_usage",
        autospec=True,
        return_value=rf.load_object("account_usage"),
    )
    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0
    assert "Usage total:" in result.output
    # The recorded month has real Speech-to-Text spend, rendered as a dollar breakdown.
    assert "Universal" in result.output
    assert "$" in result.output


def test_limits_renders_no_custom_limits(mocker):
    _with_session()
    mocker.patch(
        "aai_cli.commands.account.ams.get_rate_limits",
        autospec=True,
        return_value=rf.load_object("account_limits"),
    )
    result = runner.invoke(app, ["limits"])
    assert result.exit_code == 0
    assert "No custom rate limits" in result.output
