from __future__ import annotations

import contextlib
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

import keyring
import keyring.errors  # keyring.errors is not re-exported by keyring/__init__
import platformdirs
import tomli_w

from aai_cli.errors import NotAuthenticated

KEYRING_SERVICE = "assemblyai-cli"
ENV_API_KEY = "ASSEMBLYAI_API_KEY"
DEFAULT_PROFILE = "default"

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_profile(name: str) -> None:
    if not _PROFILE_RE.match(name):
        from aai_cli.errors import CLIError

        raise CLIError(
            f"Invalid profile name {name!r}.",
            error_type="invalid_profile",
            exit_code=2,
            suggestion="Use only letters, digits, '-' or '_'.",
        )


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir("assemblyai"))


def _config_file() -> Path:
    return config_dir() / "config.toml"


def _load() -> dict[str, Any]:
    path = _config_file()
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        try:
            data: dict[str, Any] = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            from aai_cli.errors import CLIError

            raise CLIError(
                f"Config file at {path} is not valid TOML ({exc}). Fix or delete it.",
                error_type="invalid_config",
                exit_code=2,
            ) from exc
        return data


def _dump(data: dict) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)


def get_active_profile() -> str:
    return str(_load().get("active_profile", DEFAULT_PROFILE))


def set_active_profile(name: str) -> None:
    _validate_profile(name)
    data = _load()
    data["active_profile"] = name
    data.setdefault("profiles", {}).setdefault(name, {})
    _dump(data)


def set_api_key(profile: str, api_key: str) -> None:
    _validate_profile(profile)
    keyring.set_password(KEYRING_SERVICE, profile, api_key)
    data = _load()
    data.setdefault("profiles", {}).setdefault(profile, {})
    data.setdefault("active_profile", profile)
    _dump(data)


def get_api_key(profile: str) -> str | None:
    return keyring.get_password(KEYRING_SERVICE, profile)


def get_profile_env(profile: str) -> str | None:
    """The backend environment recorded for a profile, if any (e.g. 'sandbox000')."""
    profiles = _load().get("profiles", {})
    value = profiles.get(profile, {}).get("env")
    return str(value) if value is not None else None


def set_profile_env(profile: str, env: str) -> None:
    """Bind a backend environment to a profile so its key and hosts stay matched."""
    _validate_profile(profile)
    data = _load()
    data.setdefault("profiles", {}).setdefault(profile, {})["env"] = env
    _dump(data)


def clear_api_key(profile: str) -> None:
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(KEYRING_SERVICE, profile)


SESSION_KEYRING_PREFIX = "session"  # keyring username: f"{prefix}:{profile}"


def _session_username(profile: str) -> str:
    return f"{SESSION_KEYRING_PREFIX}:{profile}"


def set_session(profile: str, *, session_jwt: str, session_token: str, account_id: int) -> None:
    """Persist the browser-login Stytch session (secret) + account id (non-secret).

    AMS self-service endpoints authenticate with this session cookie, not the API
    key. The JWT is short-lived; an expired session surfaces as NotAuthenticated.
    """
    _validate_profile(profile)
    keyring.set_password(
        KEYRING_SERVICE,
        _session_username(profile),
        json.dumps({"jwt": session_jwt, "token": session_token}),
    )
    data = _load()
    data.setdefault("profiles", {}).setdefault(profile, {})["account_id"] = account_id
    _dump(data)


def get_session(profile: str) -> dict[str, str] | None:
    """The stored {'jwt', 'token'} for a profile, or None if absent/corrupt."""
    raw = keyring.get_password(KEYRING_SERVICE, _session_username(profile))
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "jwt" not in data:
        return None
    return data


def get_account_id(profile: str) -> int | None:
    """The AMS account id recorded at login for a profile, if any."""
    value = _load().get("profiles", {}).get(profile, {}).get("account_id")
    return int(value) if value is not None else None


def clear_session(profile: str) -> None:
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(KEYRING_SERVICE, _session_username(profile))
    data = _load()
    prof = data.get("profiles", {}).get(profile)
    if prof and "account_id" in prof:
        del prof["account_id"]
        _dump(data)


def resolve_api_key(*, profile: str | None = None, api_key_flag: str | None = None) -> str:
    if api_key_flag is not None:
        if not api_key_flag:
            from aai_cli.errors import CLIError

            raise CLIError(
                "Empty --api-key provided.",
                error_type="invalid_key",
                exit_code=2,
                suggestion="Pass a non-empty key, e.g. --api-key sk_...",
            )
        return api_key_flag
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return env_key
    profile = profile or get_active_profile()
    stored = get_api_key(profile)
    if stored:
        return stored
    raise NotAuthenticated()
