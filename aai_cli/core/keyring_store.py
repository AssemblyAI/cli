"""The OS keyring as the CLI's secret store.

config.toml holds only non-secret profile settings; every secret — the API key and
the browser-login session blob — lives in the OS keyring instead, keyed under
``KEYRING_SERVICE``. This module is the single place that imports ``keyring``, so the
"secrets never touch the dotfile" boundary is structural: ``config`` reads and writes
them through this typed wrapper and never opens the keyring directly.

Every backend call is wrapped because a keyring is routinely unusable — a locked
keychain, an ACL-bound entry, or (on headless boxes) no backend at all — and those
must surface as a clean ``CLIError`` or read as "nothing stored", never a traceback.
"""

from __future__ import annotations

import base64
import contextlib
import re
import uuid

import keyring
import keyring.errors  # keyring.errors is not re-exported by keyring/__init__

from aai_cli.core.errors import CLIError

KEYRING_SERVICE = "assemblyai-cli"


def set_secret(username: str, secret: str) -> None:
    """Write a secret to the OS keyring, turning backend failures into a clean error.

    A locked keychain, or an existing entry whose ACL is bound to another app, makes
    keyring raise a KeyringError (e.g. macOS errSecInvalidOwnerEdit, -25244). Surface
    it as a CLIError so the command prints a fixable message instead of a traceback.
    """
    try:
        _write_secret(username, secret)
    except keyring.errors.KeyringError as exc:
        raise CLIError(
            f"Your OS keyring rejected the write ({exc}).",
            error_type="keyring_error",
            suggestion=(
                "Unlock your keyring, or remove the stale 'assemblyai-cli' entry and "
                "retry (macOS: security delete-generic-password -s assemblyai-cli). "
                "On a headless machine without a keyring, set ASSEMBLYAI_API_KEY instead."
            ),
        ) from exc


def get_secret(username: str) -> str | None:
    """Read a secret, treating an unusable keyring backend as "nothing stored".

    Headless machines (containers, CI, servers) routinely have no keyring backend at
    all, so keyring raises NoKeyringError on every read. That state must read as "not
    signed in" — ASSEMBLYAI_API_KEY still works there — never as a crash.
    """
    try:
        return _read_secret(username)
    except keyring.errors.KeyringError:
        return None


def restore_secret(username: str, prior: str | None) -> None:
    """Best-effort restore of a keyring entry to a snapshot value, for login rollback.

    Suppresses keyring errors (including a delete of an absent entry) so a failed
    rollback never masks the original write error that triggered it.
    """
    with contextlib.suppress(keyring.errors.KeyringError):
        if prior is None:
            _delete_secret(username)
        else:
            _write_secret(username, prior)


def delete_secret(username: str) -> None:
    """Delete a keyring entry, treating an absent entry or missing backend as success.

    KeyringError, not just PasswordDeleteError: with no backend at all (headless
    boxes) delete raises NoKeyringError, and "nothing stored" is already the goal.
    """
    with contextlib.suppress(keyring.errors.KeyringError):
        _delete_secret(username)


def usable() -> bool:
    """True when the OS keyring backend can be read.

    Headless boxes (containers, CI, bare SSH) often have no keyring backend, so
    ``keyring`` raises on every access. ``assembly doctor`` uses this to tell a user with
    no key that the *backend* is the problem — and to recommend ASSEMBLYAI_API_KEY —
    rather than pointing at `assembly login`, whose browser flow also can't persist there.
    """
    try:
        keyring.get_password(KEYRING_SERVICE, "__probe__")
    except keyring.errors.KeyringError:
        return False
    return True


# Windows generic credentials permit 2560 UTF-16 bytes. ASCII chunks stay below
# that limit, including when the original secret contains non-BMP characters.
_CHUNK_SIZE = 1000
_MAX_CREDENTIAL_BYTES = 2560
_MANIFEST = re.compile(r"assemblyai-chunks-v1:([0-9a-f]{32}):([1-9][0-9]{0,5})")
_CHUNK_SERVICE = KEYRING_SERVICE + "-chunks-v1"


def _chunk_names(raw: str | None) -> list[str]:
    match = _MANIFEST.fullmatch(raw or "")
    if not match:
        return []
    generation, count = match.groups()
    return [f"{generation}:{index}" for index in range(int(count))]


def _remove_chunks(names: list[str]) -> None:
    for name in names:
        with contextlib.suppress(keyring.errors.KeyringError):
            keyring.delete_password(_CHUNK_SERVICE, name)


def _write_secret(username: str, secret: str) -> None:
    old = keyring.get_password(KEYRING_SERVICE, username)
    names: list[str] = []
    try:
        if len(secret.encode("utf-16-le")) <= _MAX_CREDENTIAL_BYTES and not _MANIFEST.fullmatch(
            secret
        ):
            raw = secret
        else:
            encoded = base64.b64encode(secret.encode()).decode("ascii")
            generation = uuid.uuid4().hex
            for offset in range(0, len(encoded), _CHUNK_SIZE):
                name = f"{generation}:{len(names)}"
                names.append(name)
                keyring.set_password(_CHUNK_SERVICE, name, encoded[offset : offset + _CHUNK_SIZE])
            raw = f"assemblyai-chunks-v1:{generation}:{len(names)}"
        # Publish only after all chunks exist; a failed write leaves the old
        # generation readable. Each generation has independent chunk names.
        keyring.set_password(KEYRING_SERVICE, username, raw)
    except Exception:
        _remove_chunks(names)
        raise
    _remove_chunks(_chunk_names(old))


def _read_secret(username: str) -> str | None:
    raw = keyring.get_password(KEYRING_SERVICE, username)
    names = _chunk_names(raw)
    if not names:
        return raw
    chunks = [keyring.get_password(_CHUNK_SERVICE, name) for name in names]
    if any(chunk is None for chunk in chunks):
        return None
    try:
        return base64.b64decode(
            "".join(chunk for chunk in chunks if chunk is not None), validate=True
        ).decode()
    except (ValueError, UnicodeError):
        return None


def _delete_secret(username: str) -> None:
    raw = keyring.get_password(KEYRING_SERVICE, username)
    keyring.delete_password(KEYRING_SERVICE, username)
    _remove_chunks(_chunk_names(raw))
