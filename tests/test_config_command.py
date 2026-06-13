"""`assembly config` — the persisted-settings surface (path/list/get/set)."""

import json

import pytest
from typer.testing import CliRunner

from aai_cli import config
from aai_cli.errors import CLIError
from aai_cli.main import app

runner = CliRunner()


# --- config.py helpers ------------------------------------------------------


def test_list_profiles_maps_names_to_envs():
    config.set_api_key("default", "sk_1")
    config.set_api_key("staging", "sk_2")
    config.set_profile_env("staging", "sandbox000")
    assert config.list_profiles() == {"default": None, "staging": "sandbox000"}


def test_set_active_profile_switches_default():
    config.set_api_key("default", "sk_1")
    config.set_api_key("staging", "sk_2")
    config.set_active_profile("staging")
    assert config.get_active_profile() == "staging"


def test_set_active_profile_rejects_unknown_name_listing_known():
    config.set_api_key("default", "sk_1")
    with pytest.raises(CLIError) as exc:
        config.set_active_profile("nope")
    assert exc.value.error_type == "invalid_profile"
    assert exc.value.exit_code == 2
    assert "known: default" in exc.value.message
    assert exc.value.suggestion == "Create it first: assembly --profile nope login"
    assert config.get_active_profile() == "default"  # nothing was written


def test_set_active_profile_with_no_profiles_says_none_yet():
    with pytest.raises(CLIError) as exc:
        config.set_active_profile("nope")
    assert "known: none yet" in exc.value.message


def test_config_file_path_is_the_toml_under_config_dir():
    path = config.config_file_path()
    assert path.name == "config.toml"
    assert path.parent == config.config_dir()


def test_bare_config_shows_subcommand_help():
    # no_args_is_help: `assembly config` alone renders the command table instead
    # of a bare "Missing command" usage error.
    result = runner.invoke(app, ["config"])
    assert "Change one setting" in result.output  # the `set` row of the help table
    assert "Print where config.toml lives" in result.output  # the `path` row


# --- assembly config path / list ---------------------------------------------


def test_config_path_prints_bare_path():
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert result.output.strip() == str(config.config_file_path())


def test_config_path_json():
    result = runner.invoke(app, ["config", "path", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"path": str(config.config_file_path())}


def test_config_list_json_is_the_full_settings_object():
    config.set_api_key("staging", "sk_2")
    config.set_profile_env("staging", "sandbox000")
    config.set_telemetry_enabled(enabled=False)
    result = runner.invoke(app, ["config", "list", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "path": str(config.config_file_path()),
        "active_profile": "staging",
        "profiles": {"staging": "sandbox000"},
        "telemetry_enabled": False,
    }


def test_config_list_human_render_shows_rows_and_hint():
    config.set_api_key("staging", "sk_2")
    config.set_profile_env("staging", "sandbox000")
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "Config file" in result.output
    assert "Active profile" in result.output
    assert "staging (sandbox000)" in result.output  # profile listed with its env
    assert "unset" in result.output  # telemetry never chosen
    assert "assembly config set" in result.output  # the next-step hint
    assert '"path"' not in result.output  # human mode, not JSON


def test_config_list_human_with_no_profiles_says_none_yet():
    result = runner.invoke(app, ["config", "list"])
    assert result.exit_code == 0
    assert "none yet" in result.output


# --- assembly config get ------------------------------------------------------


def test_config_get_active_profile_defaults_to_default():
    result = runner.invoke(app, ["config", "get", "active_profile"])
    assert result.exit_code == 0
    assert result.output.strip() == "default"


def test_config_get_env_unset_prints_unset_and_null_json():
    human = runner.invoke(app, ["config", "get", "env"])
    assert human.exit_code == 0
    assert human.output.strip() == "unset"
    as_json = runner.invoke(app, ["config", "get", "env", "--json"])
    assert json.loads(as_json.output) == {"key": "env", "value": None}


def test_config_get_env_honors_profile_flag():
    config.set_api_key("staging", "sk_2")
    config.set_profile_env("staging", "sandbox000")
    result = runner.invoke(app, ["--profile", "staging", "config", "get", "env"])
    assert result.exit_code == 0
    assert result.output.strip() == "sandbox000"


def test_config_get_telemetry_renders_booleans_in_toml_case():
    config.set_telemetry_enabled(enabled=True)
    assert runner.invoke(app, ["config", "get", "telemetry_enabled"]).output.strip() == "true"
    config.set_telemetry_enabled(enabled=False)
    assert runner.invoke(app, ["config", "get", "telemetry_enabled"]).output.strip() == "false"


def test_config_get_rejects_unknown_key():
    result = runner.invoke(app, ["config", "get", "bogus"])
    assert result.exit_code == 2  # Typer enum validation
    assert "bogus" in result.output


# --- assembly config set ------------------------------------------------------


def test_config_set_active_profile_round_trips():
    config.set_api_key("staging", "sk_2")
    result = runner.invoke(app, ["config", "set", "active_profile", "staging"])
    assert result.exit_code == 0
    assert "active_profile = staging" in result.output
    assert config.get_active_profile() == "staging"


def test_config_set_active_profile_unknown_fails_cleanly():
    result = runner.invoke(app, ["config", "set", "active_profile", "ghost"])
    assert result.exit_code == 2
    assert "No profile named 'ghost'" in result.output


def test_config_set_env_writes_to_selected_profile():
    config.set_api_key("staging", "sk_2")
    result = runner.invoke(app, ["--profile", "staging", "config", "set", "env", "sandbox000"])
    assert result.exit_code == 0
    assert config.get_profile_env("staging") == "sandbox000"
    assert config.get_profile_env("default") is None  # only the selected profile


def test_config_set_env_rejects_unknown_environment():
    result = runner.invoke(app, ["config", "set", "env", "bogus"])
    assert result.exit_code == 2
    assert "Unknown environment 'bogus'" in result.output
    assert "production, sandbox000" in result.output


@pytest.mark.parametrize("word", ["true", "1", "yes", "on", "TRUE", " Yes "])
def test_config_set_telemetry_true_words(word):
    result = runner.invoke(app, ["config", "set", "telemetry_enabled", word])
    assert result.exit_code == 0
    assert config.get_telemetry_enabled() is True


@pytest.mark.parametrize("word", ["false", "0", "no", "off", "OFF"])
def test_config_set_telemetry_false_words(word):
    result = runner.invoke(app, ["config", "set", "telemetry_enabled", word])
    assert result.exit_code == 0
    assert config.get_telemetry_enabled() is False


def test_config_set_telemetry_rejects_non_boolean():
    result = runner.invoke(app, ["config", "set", "telemetry_enabled", "maybe"])
    assert result.exit_code == 2
    assert "telemetry_enabled expects a boolean" in result.output
    assert "'maybe'" in result.output
    assert config.get_telemetry_enabled() is None  # nothing persisted


def test_config_set_json_reports_typed_value():
    result = runner.invoke(app, ["config", "set", "telemetry_enabled", "false", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"key": "telemetry_enabled", "value": False}
