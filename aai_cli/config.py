from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from typing import Any

import keyring
import keyring.errors  # keyring.errors is not re-exported by keyring/__init__
import platformdirs
import tomli_w
import tomllib

from aai_cli.errors import NotAuthenticated

KEYRING_SERVICE = "assemblyai-cli"
ENV_API_KEY = "ASSEMBLYAI_API_KEY"
DEFAULT_PROFILE = "default"

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_profile(name: str) -> None:
    if not _PROFILE_RE.match(name):
        from aai_cli.errors import CLIError

        raise CLIError(
            f"Invalid profile name {name!r}: use letters, digits, '-' or '_' only.",
            error_type="invalid_profile",
            exit_code=2,
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
        data: dict[str, Any] = tomllib.load(fh)
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


def resolve_api_key(*, profile: str | None = None, api_key_flag: str | None = None) -> str:
    if api_key_flag is not None:
        if not api_key_flag:
            from aai_cli.errors import CLIError

            raise CLIError("Empty --api-key provided.", error_type="invalid_key", exit_code=2)
        return api_key_flag
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return env_key
    profile = profile or get_active_profile()
    stored = get_api_key(profile)
    if stored:
        return stored
    raise NotAuthenticated()
