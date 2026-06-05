from __future__ import annotations

import contextlib
import os
import re
import tempfile
import tomllib
from pathlib import Path

import keyring
import keyring.errors  # keyring.errors is not re-exported by keyring/__init__
import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_cli.errors import NotAuthenticated

KEYRING_SERVICE = "assemblyai-cli"
ENV_API_KEY = "ASSEMBLYAI_API_KEY"
DEFAULT_PROFILE = "default"

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class Profile(BaseModel):
    """A single profile's non-secret settings persisted in config.toml.

    ``extra="allow"`` so unknown keys written by a newer CLI survive a round-trip
    through an older one instead of being silently dropped on the next ``_dump``.
    """

    model_config = ConfigDict(extra="allow")

    env: str | None = None
    account_id: int | None = None


class Config(BaseModel):
    """The whole config.toml document. ``active_profile`` stays optional so we can
    tell "never set" apart from the default and only adopt a new profile as active
    when the file had none (matching the historic ``setdefault`` semantics)."""

    model_config = ConfigDict(extra="allow")

    active_profile: str | None = None
    profiles: dict[str, Profile] = Field(default_factory=dict)


class StoredSession(BaseModel):
    """The browser-login Stytch session blob persisted in the OS keyring as JSON."""

    jwt: str
    token: str = ""


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


def _load() -> Config:
    path = _config_file()
    if not path.exists():
        return Config()
    with path.open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            from aai_cli.errors import CLIError

            raise CLIError(
                f"Config file at {path} is not valid TOML ({exc}). Fix or delete it.",
                error_type="invalid_config",
                exit_code=2,
            ) from exc
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        from aai_cli.errors import CLIError

        raise CLIError(
            f"Config file at {path} has an unexpected shape ({exc}). Fix or delete it.",
            error_type="invalid_config",
            exit_code=2,
        ) from exc


def _dump(cfg: Config) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and atomically rename over the target, so a crash
    # (or concurrent reader) mid-write can never leave config.toml truncated into
    # invalid TOML that _load would then reject. os.replace is atomic within a dir.
    # exclude_none is required: TOML has no null and tomli_w rejects None values.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".toml.tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(cfg.model_dump(exclude_none=True), fh)
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def get_active_profile() -> str:
    return _load().active_profile or DEFAULT_PROFILE


def set_api_key(profile: str, api_key: str) -> None:
    _validate_profile(profile)
    keyring.set_password(KEYRING_SERVICE, profile, api_key)
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile())
    if cfg.active_profile is None:
        cfg.active_profile = profile
    _dump(cfg)


def get_api_key(profile: str) -> str | None:
    return keyring.get_password(KEYRING_SERVICE, profile)


def get_profile_env(profile: str) -> str | None:
    """The backend environment recorded for a profile, if any (e.g. 'sandbox000')."""
    prof = _load().profiles.get(profile)
    return prof.env if prof else None


def set_profile_env(profile: str, env: str) -> None:
    """Bind a backend environment to a profile so its key and hosts stay matched."""
    _validate_profile(profile)
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile()).env = env
    _dump(cfg)


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
        StoredSession(jwt=session_jwt, token=session_token).model_dump_json(),
    )
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile()).account_id = account_id
    _dump(cfg)


def get_session(profile: str) -> dict[str, str] | None:
    """The stored {'jwt', 'token'} for a profile, or None if absent/corrupt."""
    raw = keyring.get_password(KEYRING_SERVICE, _session_username(profile))
    if not raw:
        return None
    try:
        session = StoredSession.model_validate_json(raw)
    except ValidationError:
        return None
    return {"jwt": session.jwt, "token": session.token}


def get_account_id(profile: str) -> int | None:
    """The AMS account id recorded at login for a profile, if any."""
    prof = _load().profiles.get(profile)
    return prof.account_id if prof else None


def clear_session(profile: str) -> None:
    with contextlib.suppress(keyring.errors.PasswordDeleteError):
        keyring.delete_password(KEYRING_SERVICE, _session_username(profile))
    cfg = _load()
    prof = cfg.profiles.get(profile)
    if prof and prof.account_id is not None:
        prof.account_id = None
        _dump(cfg)


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
