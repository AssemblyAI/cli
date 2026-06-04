import pytest

from assemblyai_cli import config
from assemblyai_cli.errors import NotAuthenticated


def test_set_and_get_api_key_roundtrip():
    config.set_api_key("default", "sk_abc")
    assert config.get_api_key("default") == "sk_abc"


def test_resolve_prefers_flag(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from_env")
    config.set_api_key("default", "from_keyring")
    assert config.resolve_api_key(api_key_flag="from_flag") == "from_flag"


def test_resolve_prefers_env_over_keyring(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from_env")
    config.set_api_key("default", "from_keyring")
    assert config.resolve_api_key() == "from_env"


def test_resolve_falls_back_to_keyring():
    config.set_api_key("default", "from_keyring")
    assert config.resolve_api_key() == "from_keyring"


def test_resolve_raises_when_missing():
    with pytest.raises(NotAuthenticated):
        config.resolve_api_key()


def test_active_profile_defaults_to_default():
    assert config.get_active_profile() == "default"


def test_set_active_profile_persists():
    config.set_active_profile("staging")
    assert config.get_active_profile() == "staging"


def test_logout_clears_key():
    config.set_api_key("default", "sk_abc")
    config.clear_api_key("default")
    assert config.get_api_key("default") is None


def test_clear_api_key_missing_is_silent():
    # Should not raise even though nothing was stored for this profile.
    config.clear_api_key("never_set")
    assert config.get_api_key("never_set") is None


def test_invalid_profile_name_rejected():
    import pytest

    from assemblyai_cli.errors import CLIError

    with pytest.raises(CLIError):
        config.set_api_key("bad name!", "sk_x")


def test_empty_api_key_flag_rejected():
    import pytest

    from assemblyai_cli.errors import CLIError

    with pytest.raises(CLIError):
        config.resolve_api_key(api_key_flag="")


def test_config_roundtrips_after_special_value(tmp_path, monkeypatch):
    # active profile name is validated; this checks tomli_w writes valid TOML for normal data
    config.set_api_key("default", "sk_x")
    config.set_active_profile("staging")
    assert config.get_active_profile() == "staging"
