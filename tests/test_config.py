import json

import pytest

from aai_cli import config
from aai_cli.errors import CLIError, NotAuthenticated


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


def test_get_session_rejects_non_string_jwt():
    config.keyring.set_password(
        config.KEYRING_SERVICE,
        config._session_username("default"),
        json.dumps({"jwt": 123, "token": "tok"}),
    )
    assert config.get_session("default") is None


def test_resolve_raises_when_missing():
    with pytest.raises(NotAuthenticated):
        config.resolve_api_key()


def test_active_profile_defaults_to_default():
    assert config.get_active_profile() == "default"


def test_logout_clears_key():
    config.set_api_key("default", "sk_abc")
    config.clear_api_key("default")
    assert config.get_api_key("default") is None


def test_clear_api_key_missing_is_silent():
    # Should not raise even though nothing was stored for this profile.
    config.clear_api_key("never_set")
    assert config.get_api_key("never_set") is None


def test_set_api_key_keyring_failure_raises_clean_error(monkeypatch):
    # A keyring write failure (e.g. a locked/ACL-denied macOS keychain) must surface
    # as a clean CLIError, never a raw PasswordSetError traceback.
    import keyring.errors

    def boom(*_a, **_k):
        raise keyring.errors.PasswordSetError("denied by keychain")

    monkeypatch.setattr(config.keyring, "set_password", boom)
    with pytest.raises(CLIError) as exc:
        config.set_api_key("default", "sk_abc")
    assert "keyring" in exc.value.message.lower()
    assert exc.value.suggestion is not None


def test_set_session_keyring_failure_raises_clean_error(monkeypatch):
    import keyring.errors

    def boom(*_a, **_k):
        raise keyring.errors.PasswordSetError("denied by keychain")

    monkeypatch.setattr(config.keyring, "set_password", boom)
    with pytest.raises(CLIError) as exc:
        config.set_session("default", session_jwt="j", session_token="t", account_id=1)
    assert "keyring" in exc.value.message.lower()
    assert exc.value.suggestion is not None


def test_persist_login_writes_key_env_and_session():
    config.persist_login(
        "default",
        api_key="sk_new",
        env="production",
        session_jwt="j",
        session_token="t",
        account_id=7,
    )
    assert config.get_api_key("default") == "sk_new"
    assert config.get_profile_env("default") == "production"
    assert config.get_account_id("default") == 7
    assert config.get_session("default") == {"jwt": "j", "token": "t"}


def _fail_on_session_write(monkeypatch):
    """Make keyring writes to the session entry fail, leaving the api-key write alone.

    Reproduces the partial-write window: set_api_key succeeds, set_session does not.
    """
    import keyring.errors

    real_set = config.keyring.set_password

    def selective(service, username, secret):
        if username.startswith(config.SESSION_KEYRING_PREFIX + ":"):
            raise keyring.errors.PasswordSetError("denied by keychain")
        return real_set(service, username, secret)

    monkeypatch.setattr(config.keyring, "set_password", selective)


def test_persist_login_rolls_back_to_empty_when_session_write_fails(monkeypatch):
    # A fresh profile: if the session write fails, nothing must persist — no orphaned
    # API key or env that would make the CLI look logged in with no usable session.
    _fail_on_session_write(monkeypatch)
    with pytest.raises(CLIError):
        config.persist_login(
            "default",
            api_key="sk_new",
            env="production",
            session_jwt="j",
            session_token="t",
            account_id=5,
        )
    assert config.get_api_key("default") is None
    assert config.get_profile_env("default") is None
    assert config.get_account_id("default") is None
    assert config.get_session("default") is None


def test_persist_login_restores_prior_credentials_when_session_write_fails(monkeypatch):
    # An existing logged-in profile must be left exactly as it was if a re-login fails
    # partway, rather than clobbered with the half-applied new values.
    config.set_api_key("default", "sk_old")
    config.set_profile_env("default", "sandbox000")
    config.set_session("default", session_jwt="oldj", session_token="oldt", account_id=1)

    _fail_on_session_write(monkeypatch)
    with pytest.raises(CLIError):
        config.persist_login(
            "default",
            api_key="sk_new",
            env="production",
            session_jwt="j",
            session_token="t",
            account_id=5,
        )
    assert config.get_api_key("default") == "sk_old"
    assert config.get_profile_env("default") == "sandbox000"
    assert config.get_account_id("default") == 1
    assert config.get_session("default") == {"jwt": "oldj", "token": "oldt"}


def test_invalid_profile_name_rejected():
    import pytest

    from aai_cli.errors import CLIError

    with pytest.raises(CLIError):
        config.set_api_key("bad name!", "sk_x")


def test_empty_api_key_flag_rejected():
    import pytest

    from aai_cli.errors import CLIError

    with pytest.raises(CLIError):
        config.resolve_api_key(api_key_flag="")


def test_invalid_profile_name_has_suggestion():
    with pytest.raises(CLIError) as exc:
        config.set_api_key("bad name!", "sk_x")
    assert exc.value.message.startswith("Invalid profile name")
    assert exc.value.suggestion == "Use only letters, digits, '-' or '_'."


def test_malformed_config_raises_clean_error(tmp_config):
    from aai_cli.errors import CLIError

    (tmp_config / "config.toml").write_text("this is not = = valid toml ===\n")
    with pytest.raises(CLIError):
        config.get_active_profile()


def test_unexpected_config_shape_raises_clean_error(tmp_config):
    # Valid TOML but the wrong shape (profiles must be a table, not a string) is
    # surfaced as a clean invalid_config CLIError, not a pydantic traceback.
    (tmp_config / "config.toml").write_text('profiles = "oops"\n')
    with pytest.raises(CLIError) as exc:
        config.get_active_profile()
    assert exc.value.error_type == "invalid_config"


def test_config_roundtrips_after_special_value(tmp_path, monkeypatch):
    # profile names are validated; this checks tomli_w writes valid TOML for normal data
    config.set_api_key("staging", "sk_x")
    assert config.get_active_profile() == "staging"


def test_dump_is_atomic_and_leaves_no_temp_files(tmp_config):
    # Atomic write: config.toml is the only file in the dir after writes (no leftover
    # .config-*.toml.tmp), and the data is intact valid TOML.
    config.set_api_key("default", "sk_abc")
    config.set_profile_env("default", "sandbox000")
    names = sorted(p.name for p in tmp_config.iterdir())
    assert names == ["config.toml"]
    assert config.get_profile_env("default") == "sandbox000"


def test_dump_cleans_up_temp_file_when_write_fails(tmp_config, monkeypatch):
    # A failure mid-write must propagate AND leave no temp file behind, so the real
    # config.toml is never clobbered by a partial write.
    config.set_api_key("default", "sk_abc")  # establish a valid config.toml first

    def boom(_data, _fh):
        raise RuntimeError("disk full")

    monkeypatch.setattr(config.tomli_w, "dump", boom)
    with pytest.raises(RuntimeError):
        config._dump(config.Config())

    names = sorted(p.name for p in tmp_config.iterdir())
    assert names == ["config.toml"]  # no .config-*.toml.tmp left behind
    assert config.get_api_key("default") == "sk_abc"  # original untouched


def test_load_caches_parsed_config_between_calls(monkeypatch):
    # One CLI invocation reads the config several times (profile, env, key); the
    # parse must happen once, with later reads served from the mtime-keyed cache.
    config.set_profile_env("default", "production")
    parses = {"n": 0}
    real_load = config.tomllib.load

    def counting_load(fh):
        parses["n"] += 1
        return real_load(fh)

    monkeypatch.setattr(config.tomllib, "load", counting_load)
    assert config.get_active_profile() == "default"
    assert config.get_profile_env("default") == "production"
    assert parses["n"] == 1


def test_write_invalidates_load_cache(monkeypatch):
    # A _dump must never leave a stale parse behind: the next read reflects it.
    config.set_profile_env("default", "production")
    assert config.get_profile_env("default") == "production"  # primes the cache
    config.set_profile_env("default", "sandbox000")
    assert config.get_profile_env("default") == "sandbox000"


def test_load_returns_independent_copies():
    # Callers mutate the returned Config (persist_login snapshots one for rollback),
    # so the cache must hand out copies — a caller's mutation can't poison it.
    config.set_api_key("default", "sk_abc")
    first = config._load()  # cache miss: parses and primes the cache
    first.active_profile = "mutated"
    first.profiles.clear()
    assert config._load().active_profile == "default"
    hit = config._load()  # cache hit: must also be a *deep* copy
    hit.profiles.clear()
    assert "default" in config._load().profiles
