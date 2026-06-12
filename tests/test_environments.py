import pytest

from aai_cli import config, environments
from aai_cli.errors import CLIError


def test_get_returns_named_environment():
    env = environments.get("sandbox000")
    assert env.name == "sandbox000"
    assert env.streaming_host == "streaming.sandbox000.assemblyai-labs.com"
    assert env.api_base == "https://api.sandbox000.assemblyai-labs.com"
    assert env.llm_gateway_base == "https://llm-gateway.sandbox000.assemblyai-labs.com/v1"


def test_production_uses_production_ams_endpoint():
    env = environments.get("production")
    assert env.ams_base == "https://ams.internal.assemblyai-labs.com"


def test_sync_base_per_environment():
    assert environments.get("production").sync_base == "https://sync.assemblyai.com"
    assert environments.get("sandbox000").sync_base == "https://sync.sandbox000.assemblyai-labs.com"


def test_get_unknown_raises_cli_error():
    with pytest.raises(CLIError) as exc:
        environments.get("nope")
    assert exc.value.exit_code == 2
    # Carries a recovery hint covering all three sources of the name (flag/env/profile).
    assert exc.value.suggestion is not None
    assert "--env" in exc.value.suggestion
    assert "unset AAI_ENV" in exc.value.suggestion
    # A bad env stored in the profile makes every command fail, and neither --env nor
    # AAI_ENV is the fix there — the hint must point at the profile's config.toml too.
    assert "profile's stored env in config.toml" in exc.value.suggestion


def test_resolve_precedence(monkeypatch):
    monkeypatch.delenv("AAI_ENV", raising=False)
    # default when nothing is provided
    assert environments.resolve(None, None).name == environments.DEFAULT_ENV
    # profile-stored env beats default
    assert environments.resolve(None, "production").name == "production"
    # AAI_ENV beats the profile
    monkeypatch.setenv("AAI_ENV", "sandbox000")
    assert environments.resolve(None, "production").name == "sandbox000"
    # explicit flag beats everything
    assert environments.resolve("production", "sandbox000").name == "production"


def test_set_active_and_active():
    environments.set_active(environments.get("production"))
    try:
        assert environments.active().name == "production"
    finally:
        environments.set_active(environments.get(environments.DEFAULT_ENV))


def test_profile_env_roundtrip():
    assert config.get_profile_env("default") is None
    config.set_profile_env("default", "sandbox000")
    assert config.get_profile_env("default") == "sandbox000"


def test_sandbox_has_streaming_tts_host():
    assert (
        environments.get("sandbox000").streaming_tts_host
        == "streaming-tts.sandbox000.assemblyai-labs.com"
    )


def test_production_has_no_streaming_tts_host():
    # Empty host is the "TTS not available here" signal the speak command checks.
    assert environments.get("production").streaming_tts_host == ""
