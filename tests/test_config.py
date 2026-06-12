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


def test_keyring_usable_true_with_a_working_backend():
    # The autouse MemoryKeyring backend reads fine, so the probe reports usable.
    assert config.keyring_usable() is True


def test_keyring_usable_false_when_backend_raises(monkeypatch):
    import keyring.errors

    def boom(*_a, **_k):
        raise keyring.errors.NoKeyringError("no backend on this headless box")

    monkeypatch.setattr(config.keyring, "get_password", boom)
    assert config.keyring_usable() is False


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

    with pytest.raises(CLIError) as exc:
        config.resolve_api_key(api_key_flag="")
    assert exc.value.error_type == "invalid_key"
    assert exc.value.exit_code == 2  # usage error, not the generic 1


def test_invalid_profile_name_has_suggestion():
    with pytest.raises(CLIError) as exc:
        config.set_api_key("bad name!", "sk_x")
    assert exc.value.message.startswith("Invalid profile name")
    assert exc.value.error_type == "invalid_profile"
    assert exc.value.exit_code == 2  # usage error, not the generic 1
    assert exc.value.suggestion == "Use only letters, digits, '-' or '_'."


def test_malformed_config_raises_clean_error(tmp_config):
    from aai_cli.errors import CLIError

    (tmp_config / "config.toml").write_text("this is not = = valid toml ===\n")
    with pytest.raises(CLIError) as exc:
        config.get_active_profile()
    assert exc.value.error_type == "invalid_config"
    assert exc.value.exit_code == 2  # usage error, not the generic 1


def test_unexpected_config_shape_raises_clean_error(tmp_config):
    # Valid TOML but the wrong shape (profiles must be a table, not a string) is
    # surfaced as a clean invalid_config CLIError, not a pydantic traceback.
    (tmp_config / "config.toml").write_text('profiles = "oops"\n')
    with pytest.raises(CLIError) as exc:
        config.get_active_profile()
    assert exc.value.error_type == "invalid_config"
    assert exc.value.exit_code == 2  # usage error, not the generic 1


def test_unexpected_config_shape_error_is_compact(tmp_config):
    # The message keeps the failing field + a short reason and stays actionable,
    # without pydantic's doc URLs or a dump of the offending input value.
    (tmp_config / "config.toml").write_text('profiles = "oops"\n')
    with pytest.raises(CLIError) as exc:
        config.get_active_profile()
    message = exc.value.message
    assert "profiles:" in message  # the field that failed
    assert "Fix or delete it." in message  # the actionable next step
    assert "errors.pydantic.dev" not in message  # no pydantic doc URLs
    assert "input_value" not in message  # no raw input dump
    assert "oops" not in message  # the bad value itself isn't echoed back


def test_unexpected_config_shape_error_names_nested_field(tmp_config):
    (tmp_config / "config.toml").write_text("[profiles.default]\nenv = 12\n")
    with pytest.raises(CLIError) as exc:
        config.get_active_profile()
    # The dotted location pinpoints which profile key is wrong.
    assert "profiles.default.env:" in exc.value.message


def test_validation_summary_joins_multiple_problems():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        config.Config.model_validate({"profiles": "oops", "active_profile": 7})
    summary = config._validation_summary(exc.value)
    assert "profiles:" in summary
    assert "active_profile:" in summary
    assert "; " in summary  # both problems, compactly joined
    assert "https://" not in summary


def test_validation_summary_labels_rootlevel_problems():
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        config.Config.model_validate("not a table")
    assert config._validation_summary(exc.value).startswith("top level: ")


def test_dump_creates_missing_parent_directories(monkeypatch, tmp_path):
    # The config dir's parents may not exist yet (first run on a fresh machine);
    # _dump must create the whole chain (mkdir parents=True), not just the leaf.
    nested = tmp_path / "deeply" / "nested" / "config"
    monkeypatch.setattr("aai_cli.config.config_dir", lambda: nested)
    config.set_api_key("default", "sk_abc")
    assert nested.is_dir()
    assert (nested / "config.toml").exists()
    assert config.get_api_key("default") == "sk_abc"


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


def test_get_api_key_treats_broken_keyring_backend_as_no_key(monkeypatch):
    # Headless boxes (containers, CI) have no keyring backend at all; reads raise
    # NoKeyringError there. That must read as "not signed in", never as a crash.
    import keyring
    import keyring.errors

    def no_backend(service, username):
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "get_password", no_backend)
    assert config.get_api_key("default") is None
    assert config.get_session("default") is None
    with pytest.raises(NotAuthenticated):
        config.resolve_api_key()


def test_resolve_api_key_env_var_works_without_keyring_backend(monkeypatch):
    import keyring
    import keyring.errors

    def no_backend(service, username):
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "get_password", no_backend)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "from_env")
    assert config.resolve_api_key() == "from_env"


def test_clear_credentials_without_keyring_backend_is_silent(monkeypatch):
    import keyring
    import keyring.errors

    def no_backend(service, username):
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "delete_password", no_backend)
    config.clear_api_key("default")
    config.clear_session("default")


def test_keyring_write_failure_suggestion_covers_headless_linux(monkeypatch):
    # The macOS keychain advice is useless on a headless Linux box; the suggestion
    # must also offer the ASSEMBLYAI_API_KEY env-var path that works everywhere.
    import keyring
    import keyring.errors

    def rejected(service, username, secret):
        raise keyring.errors.KeyringError("locked")

    monkeypatch.setattr(keyring, "set_password", rejected)
    with pytest.raises(CLIError) as exc:
        config.set_api_key("default", "sk_x")
    assert exc.value.error_type == "keyring_error"
    assert exc.value.suggestion is not None
    assert "set ASSEMBLYAI_API_KEY instead" in exc.value.suggestion
    # The macOS path stays for keychain users.
    assert "security delete-generic-password -s assemblyai-cli" in exc.value.suggestion


def test_resolve_treats_whitespace_only_env_key_as_missing(monkeypatch):
    # `export ASSEMBLYAI_API_KEY='   '` must read as "no key" (the clean exit-4
    # not-signed-in path), never reach httpx as an illegal header value.
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "   ")
    with pytest.raises(NotAuthenticated):
        config.resolve_api_key()


def test_resolve_strips_padding_from_env_key(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "  sk_padded \n")
    assert config.resolve_api_key() == "sk_padded"


def test_resolve_treats_whitespace_only_keyring_key_as_missing():
    config.set_api_key("default", "   ")
    with pytest.raises(NotAuthenticated):
        config.resolve_api_key()


def test_resolve_strips_padding_from_keyring_key():
    config.set_api_key("default", " sk_stored\n")
    assert config.resolve_api_key() == "sk_stored"


def test_resolve_rejects_whitespace_only_flag():
    with pytest.raises(CLIError) as exc:
        config.resolve_api_key(api_key_flag="   ")
    assert exc.value.error_type == "invalid_key"
    assert exc.value.exit_code == 2


def test_resolve_strips_padding_from_flag():
    assert config.resolve_api_key(api_key_flag=" sk_flag ") == "sk_flag"


def test_validate_profile_public_wrapper():
    # Public so resolution-time callers (context.AppState) can fail fast on a typo'd
    # --profile before any network work.
    config.validate_profile("ok-name_1")  # valid: no exception
    with pytest.raises(CLIError) as exc:
        config.validate_profile("bad name!")
    assert exc.value.exit_code == 2
    assert exc.value.message.startswith("Invalid profile name")
