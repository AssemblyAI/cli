"""The shared ``--list-voices`` flag of the streaming-TTS commands (speak, dub, live):
one eager Typer option (aai_cli/options.py) over one body (aai_cli/app/tts_voices.py),
so the three are driven together here."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aai_cli.core import config
from aai_cli.main import app
from aai_cli.tts import voice_list
from tests._tts_session_helpers import use_no_tts_production

# Most cases run under --sandbox, which the root callback gates on an AssemblyAI
# login — run the whole module as an employee.
pytestmark = pytest.mark.usefixtures("internal_profile")

runner = CliRunner()

COMMANDS = ["speak", "dub", "live"]
# The account's own voices, as GET /v1/voices returns them (abridged).
VOICES = [{"slug": "alex-6f2f2e", "name": "Alex", "language": "english", "visibility": "private"}]


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "test-key")
    seen: dict[str, object] = {}

    def _fetch(api_key: str):
        seen["api_key"] = api_key
        return VOICES

    monkeypatch.setattr(voice_list, "fetch_voices", _fetch)
    return seen


@pytest.mark.parametrize("command", COMMANDS)
def test_lists_the_service_voices(command, fake_fetch):
    # No other argument/option is needed — notably `dub` needs no --lang, because the
    # eager flag exits before the required option is checked.
    result = runner.invoke(app, ["--sandbox", command, "--list-voices"])
    assert result.exit_code == 0, result.output
    assert fake_fetch["api_key"] == "test-key"
    # The built-in voices first, then the account's own; slugs padded to the widest.
    assert "aai-jane     Jane · built-in, speaks every language" in result.output
    assert "alex-6f2f2e  Alex · english" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_json_emits_the_raw_listing(command, fake_fetch):
    # --json is honored even though it parses after the eager flag: the server's
    # records are dumped untouched (unknown fields like "visibility" survive).
    result = runner.invoke(app, ["--sandbox", command, "--list-voices", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == voice_list.builtin_voices() + VOICES
    assert "Alex · english" not in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_lists_in_production(command, fake_fetch):
    result = runner.invoke(app, [command, "--list-voices"])  # default = production
    assert result.exit_code == 0, result.output
    assert "alex-6f2f2e  Alex · english" in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_env_without_tts_host_is_rejected_with_the_command_named(command, fake_fetch, monkeypatch):
    use_no_tts_production(monkeypatch)
    result = runner.invoke(app, [command, "--list-voices"])
    assert result.exit_code == 2
    assert f"assembly {command} needs streaming TTS" in " ".join(result.output.split())
    assert "api_key" not in fake_fetch  # refused before any request


def test_wins_over_an_invalid_earlier_option(fake_fetch):
    # Like --help, the flag is eager: it is processed before the options that precede
    # it on the line, so an invalid one (--sample-rate must be positive) never gets
    # to reject the invocation.
    result = runner.invoke(app, ["--sandbox", "speak", "--sample-rate", "0", "--list-voices"])
    assert result.exit_code == 0, result.output
    assert "aai-jane" in result.output


def test_flag_off_runs_the_command_normally(fake_fetch):
    # Without the flag the eager callback is a pass-through: the command's own
    # validation runs (dub's required --lang is reported, no listing is fetched).
    result = runner.invoke(app, ["--sandbox", "dub", "talk.mp4"])
    assert result.exit_code == 2
    assert "api_key" not in fake_fetch
