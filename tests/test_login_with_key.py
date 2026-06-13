"""`assembly login --with-api-key` (stdin key) and the `--api-key` deprecation trap."""

import json
import types

import pytest
from typer.testing import CliRunner

from aai_cli.commands import login as login_cmd
from aai_cli.core import config
from aai_cli.core.errors import STDIN_KEY_RECIPE, UsageError
from aai_cli.main import app
from tests._snapshot_surface import normalize

runner = CliRunner()


def test_stdin_key_recipe_is_the_documented_pipe():
    # The constant is interpolated into help, errors, and the auth flow; pin it once.
    assert STDIN_KEY_RECIPE == "printenv ASSEMBLYAI_API_KEY | assembly login --with-api-key"


def test_with_api_key_reads_stdin_and_stores(mocker):
    validate = mocker.patch(
        "aai_cli.commands.login.client.validate_key", autospec=True, return_value=True
    )
    result = runner.invoke(app, ["login", "--with-api-key", "--json"], input="  sk_piped \n")
    assert result.exit_code == 0
    assert config.get_api_key("default") == "sk_piped"  # whitespace stripped
    validate.assert_called_once_with("sk_piped")  # the stdin key is what's validated
    objs = [json.loads(line) for line in result.output.strip().splitlines()]
    payload = next(o for o in objs if "authenticated" in o)
    assert payload["api_key_only"] is True  # no AMS session from a key-only login


def test_with_api_key_human_mode_mentions_account_command_limit(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--with-api-key"], input="sk_piped\n")
    assert result.exit_code == 0
    assert "Signed in as default" in result.output
    assert "browser flow" in result.output  # the key-only caveat hint


def test_with_api_key_empty_stdin_fails_with_recipe(mocker):
    validate = mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True)
    result = runner.invoke(app, ["login", "--with-api-key"], input="   \n")
    assert result.exit_code == 2
    assert "found no key on stdin" in result.output
    assert "printenv ASSEMBLYAI_API_KEY" in result.output
    validate.assert_not_called()
    assert config.get_api_key("default") is None


def test_read_stdin_key_rejects_a_terminal_stdin(monkeypatch):
    # A TTY stdin means nothing was piped; reading would block forever.
    monkeypatch.setattr(
        "sys.stdin", types.SimpleNamespace(isatty=lambda: True, read=lambda: "never")
    )
    with pytest.raises(UsageError) as exc:
        login_cmd._read_stdin_key()
    assert "stdin is a terminal" in exc.value.message
    assert STDIN_KEY_RECIPE in (exc.value.suggestion or "")


def test_api_key_and_with_api_key_conflict(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_x", "--with-api-key"], input="sk_y\n")
    assert result.exit_code == 2
    assert "--api-key and --with-api-key can't be combined" in result.output
    assert config.get_api_key("default") is None


def test_api_key_flag_warns_toward_stdin_form(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert "shell history" in result.output  # the deprecation warning
    assert "--with-api-key" in result.output
    assert config.get_api_key("default") == "sk_flag"  # but it still works


def test_api_key_warning_respects_quiet(mocker):
    mocker.patch("aai_cli.commands.login.client.validate_key", autospec=True, return_value=True)
    result = runner.invoke(app, ["--quiet", "login", "--api-key", "sk_flag"])
    assert result.exit_code == 0
    assert "shell history" not in result.output


def test_api_key_flag_is_hidden_from_help():
    result = runner.invoke(app, ["login", "--help"])
    assert result.exit_code == 0
    # Strip Rich's ANSI styling first: CI forces color (FORCE_COLOR), which interleaves
    # escape codes inside the rendered option name, so a raw substring check would miss it.
    help_text = normalize(result.output)
    assert "--with-api-key" in help_text
    assert "--api-key " not in help_text  # trailing space: --with-api-key still matches
