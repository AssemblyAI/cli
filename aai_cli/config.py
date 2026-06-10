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

from aai_cli.errors import CLIError, NotAuthenticated

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


# Parsed-config cache: path -> (mtime_ns, size, parsed). The several _load()
# calls in one CLI invocation (profile, env, key resolution) then don't each
# re-read and re-parse the same unchanged TOML; _dump() bumps the mtime, which
# invalidates it naturally. Callers mutate the returned Config (and persist_login
# snapshots one for rollback), so hand out deep copies, never the cached object.
_load_cache: dict[Path, tuple[int, int, Config]] = {}


def _load() -> Config:
    path = _config_file()
    try:
        stat = path.stat()
    except OSError:
        return Config()
    cached = _load_cache.get(path)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2].model_copy(deep=True)
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
        cfg = Config.model_validate(data)
    except ValidationError as exc:
        from aai_cli.errors import CLIError

        raise CLIError(
            f"Config file at {path} has an unexpected shape ({exc}). Fix or delete it.",
            error_type="invalid_config",
            exit_code=2,
        ) from exc
    _load_cache[path] = (stat.st_mtime_ns, stat.st_size, cfg)
    return cfg.model_copy(deep=True)


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
        # The mtime/size key usually invalidates on its own, but drop the entry
        # explicitly so a same-size rewrite on a coarse-mtime filesystem can't
        # serve the pre-write parse.
        _load_cache.pop(path, None)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def get_active_profile() -> str:
    return _load().active_profile or DEFAULT_PROFILE


def _keyring_set(username: str, secret: str) -> None:
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
                "retry (macOS: security delete-generic-password -s assemblyai-cli)."
            ),
        ) from exc


def _keyring_restore(username: str, prior: str | None) -> None:
    """Best-effort restore of a keyring entry to a snapshot value, for login rollback.

    Suppresses keyring errors (including a delete of an absent entry) so a failed
    rollback never masks the original write error that triggered it.
    """
    with contextlib.suppress(keyring.errors.KeyringError):
        if prior is None:
            keyring.delete_password(KEYRING_SERVICE, username)
        else:
            keyring.set_password(KEYRING_SERVICE, username, prior)


def set_api_key(profile: str, api_key: str) -> None:
    _validate_profile(profile)
    _keyring_set(profile, api_key)
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
    _keyring_set(
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


def persist_login(
    profile: str,
    *,
    api_key: str,
    env: str,
    session_jwt: str,
    session_token: str,
    account_id: int,
) -> None:
    """Atomically persist a full browser-login result (API key + env + session).

    The three writes span the keyring and config.toml, so a mid-sequence failure
    (e.g. a locked keychain after the key is already stored) would otherwise leave a
    half-written profile — an API key with no session, which looks signed-in but
    can't reach AMS. On any failure the pre-login snapshot is restored: config.toml
    is rewritten verbatim in one atomic dump, and the two keyring entries are
    restored best-effort.
    """
    _validate_profile(profile)
    prior_api_key = keyring.get_password(KEYRING_SERVICE, profile)
    prior_session = keyring.get_password(KEYRING_SERVICE, _session_username(profile))
    prior_cfg = _load()
    done = False
    try:
        set_api_key(profile, api_key)
        set_profile_env(profile, env)
        set_session(
            profile,
            session_jwt=session_jwt,
            session_token=session_token,
            account_id=account_id,
        )
        done = True
    finally:
        if not done:
            _keyring_restore(profile, prior_api_key)
            _keyring_restore(_session_username(profile), prior_session)
            _dump(prior_cfg)


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


def resolve_api_key_optional(*, profile: str | None = None) -> str | None:
    """The same key chain as ``resolve_api_key`` (env -> keyring), but ``None`` instead
    of raising when no key is configured — for callers that work without one
    (``aai init`` scaffolding, the onboarding wizard's signed-in check)."""
    try:
        return resolve_api_key(profile=profile)
    except NotAuthenticated:
        return None
