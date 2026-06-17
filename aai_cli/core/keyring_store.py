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

import contextlib

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
        keyring.set_password(KEYRING_SERVICE, username, secret)
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
        return keyring.get_password(KEYRING_SERVICE, username)
    except keyring.errors.KeyringError:
        return None


def restore_secret(username: str, prior: str | None) -> None:
    """Best-effort restore of a keyring entry to a snapshot value, for login rollback.

    Suppresses keyring errors (including a delete of an absent entry) so a failed
    rollback never masks the original write error that triggered it.
    """
    with contextlib.suppress(keyring.errors.KeyringError):
        if prior is None:
            keyring.delete_password(KEYRING_SERVICE, username)
        else:
            keyring.set_password(KEYRING_SERVICE, username, prior)


def delete_secret(username: str) -> None:
    """Delete a keyring entry, treating an absent entry or missing backend as success.

    KeyringError, not just PasswordDeleteError: with no backend at all (headless
    boxes) delete raises NoKeyringError, and "nothing stored" is already the goal.
    """
    with contextlib.suppress(keyring.errors.KeyringError):
        keyring.delete_password(KEYRING_SERVICE, username)


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
