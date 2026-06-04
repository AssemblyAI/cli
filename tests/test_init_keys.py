from aai_cli import config
from aai_cli.init import keys


def test_resolves_from_env(monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "env-key-123")
    assert keys.resolve_optional_api_key(profile=None) == "env-key-123"


def test_resolves_from_keyring(memory_keyring):
    config.set_api_key("default", "stored-key-456")
    assert keys.resolve_optional_api_key(profile=None) == "stored-key-456"


def test_returns_none_when_absent():
    # isolate_env strips the env var and memory_keyring starts empty.
    assert keys.resolve_optional_api_key(profile=None) is None
