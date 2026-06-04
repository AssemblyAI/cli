import os

import keyring
import pytest
from keyring.backend import KeyringBackend

# Captured at import, before `isolate_env` strips ASSEMBLYAI_API_KEY from the
# environment. The e2e suite uses this real key to drive the CLI as a subprocess;
# unit tests still run fully isolated.
REAL_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY")


@pytest.fixture
def real_api_key():
    """The real API key from the environment, or skip if none is set."""
    if not REAL_API_KEY:
        pytest.skip("ASSEMBLYAI_API_KEY not set; skipping real-API e2e test.")
    return REAL_API_KEY


class MemoryKeyring(KeyringBackend):
    # A plain value is the documented way to set a backend's priority.
    priority = 1

    def __init__(self):
        self._store = {}

    def get_password(self, service, username):
        return self._store.get((service, username))

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def delete_password(self, service, username):
        if (service, username) not in self._store:
            import keyring.errors

            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    for var in ("ASSEMBLYAI_API_KEY", "CI", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def memory_keyring():
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    return backend


@pytest.fixture(autouse=True)
def tmp_config(monkeypatch, tmp_path):
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    monkeypatch.setattr("aai_cli.config.config_dir", lambda: cfg_dir)
    return cfg_dir
