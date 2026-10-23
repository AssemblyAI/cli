from __future__ import annotations

import keyring
import keyring.errors
import pytest

from aai_cli.core import keyring_store


def test_large_session_fits_windows_credential_limit(monkeypatch):
    original = keyring.set_password

    def limited(service: str, username: str, password: str) -> None:
        if len(password.encode("utf-16-le")) > 2560:
            raise ValueError("CredWrite: credential exceeds Windows limit")
        original(service, username, password)

    monkeypatch.setattr(keyring, "set_password", limited)
    secret = "fake-session-😀" * 400
    keyring_store.set_secret("session:default", secret)
    assert keyring_store.get_secret("session:default") == secret
    keyring_store.delete_secret("session:default")
    assert keyring_store.get_secret("session:default") is None


def test_failed_large_write_preserves_old_secret(monkeypatch):
    keyring_store.set_secret("session:default", "old")
    original = keyring.set_password
    calls = 0

    def fail_second(service: str, username: str, password: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise keyring.errors.PasswordSetError("locked")
        original(service, username, password)

    monkeypatch.setattr(keyring, "set_password", fail_second)
    with pytest.raises(keyring_store.CLIError):
        keyring_store.set_secret("session:default", "x" * 4000)
    assert keyring_store.get_secret("session:default") == "old"


@pytest.mark.parametrize("replacement", ["small", "y" * 5000])
def test_overwrite_removes_old_chunks(replacement):
    keyring_store.set_secret("session:default", "x" * 4000)
    raw = keyring.get_password(keyring_store.KEYRING_SERVICE, "session:default")
    old_names = keyring_store._chunk_names(raw)
    assert old_names
    keyring_store.set_secret("session:default", replacement)
    assert keyring_store.get_secret("session:default") == replacement
    assert all(keyring.get_password(keyring_store._CHUNK_SERVICE, n) is None for n in old_names)


@pytest.mark.parametrize("corrupt", [None, "!invalid-base64!", "/w=="])
def test_missing_or_corrupt_chunk_is_not_a_session(corrupt):
    keyring_store.set_secret("session:default", "x" * 4000)
    raw = keyring.get_password(keyring_store.KEYRING_SERVICE, "session:default")
    name = keyring_store._chunk_names(raw)[0]
    if corrupt is None:
        keyring.delete_password(keyring_store._CHUNK_SERVICE, name)
    else:
        keyring.set_password(keyring_store._CHUNK_SERVICE, name, corrupt)
    assert keyring_store.get_secret("session:default") is None


def test_restore_and_delete_large_secret():
    secret = "z" * 4000
    keyring_store.restore_secret("session:default", secret)
    assert keyring_store.get_secret("session:default") == secret
    raw = keyring.get_password(keyring_store.KEYRING_SERVICE, "session:default")
    names = keyring_store._chunk_names(raw)
    keyring_store.restore_secret("session:default", None)
    assert keyring_store.get_secret("session:default") is None
    assert all(keyring.get_password(keyring_store._CHUNK_SERVICE, n) is None for n in names)


def test_manifest_shaped_secret_round_trips():
    secret = "assemblyai-chunks-v1:" + "a" * 32 + ":1"
    keyring_store.set_secret("default", secret)
    assert keyring_store.get_secret("default") == secret
