from __future__ import annotations

import contextlib
import os
import re
import tempfile
import tomllib
import uuid
from pathlib import Path

import keyring
import keyring.errors  # keyring.errors is not re-exported by keyring/__init__
import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_cli.core import debuglog, env
from aai_cli.core.errors import CLIError, NotAuthenticated

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
    # Login email from AMS discovery; gates internal-environment access (see core.access).
    email: str | None = None


class Config(BaseModel):
    """The whole config.toml document. ``active_profile`` stays optional so we can
    tell "never set" apart from the default and only adopt a new profile as active
    when the file had none (matching the historic ``setdefault`` semantics)."""

    model_config = ConfigDict(extra="allow")

    active_profile: str | None = None
    profiles: dict[str, Profile] = Field(default_factory=dict)
    # Telemetry state (see telemetry.py): a random anonymous install id, and the
    # persisted opt-out. None means "never chosen", which the opt-out model reads
    # as enabled — distinct from an explicit False written by `assembly telemetry disable`.
    device_id: str | None = None
    telemetry_enabled: bool | None = None
    update_last_check: float | None = None
    update_latest_version: str | None = None


class StoredSession(BaseModel):
    """The browser-login Stytch session blob persisted in the OS keyring as JSON."""

    jwt: str
    token: str = ""


def validate_profile(name: str) -> None:
    """Reject profile names that aren't simple identifiers.

    Public so resolution-time callers (``context.AppState.resolve_profile``) can
    fail fast on a typo'd ``--profile`` before any network work, instead of only
    tripping over it at keyring-write time.
    """
    if not _PROFILE_RE.match(name):
        raise CLIError(
            f"Invalid profile name {name!r}.",
            error_type="invalid_profile",
            exit_code=2,
            suggestion="Use only letters, digits, '-' or '_'.",
        )


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir("assemblyai"))


def config_file_path() -> Path:
    """Where config.toml lives — surfaced by `assembly config path` so users can
    find the file without knowing the platformdirs convention."""
    return _config_file()


def _config_file() -> Path:
    return config_dir() / "config.toml"


def _validation_summary(exc: ValidationError) -> str:
    """A compact, human-sized summary of a pydantic ValidationError.

    Just "field: reason" per problem — pydantic's full rendering dumps input values
    and errors.pydantic.dev doc URLs, which is noise (and a potential value leak)
    in a one-line CLI error.
    """
    problems: list[str] = []
    # include_url/include_input=False keep pydantic's url/input fields out of each
    # error dict, but this summary only reads loc + msg, so flipping them is an
    # equivalent mutant (the rendered string is identical either way).
    for err in exc.errors(include_url=False, include_input=False):  # pragma: no mutate
        loc = ".".join(str(part) for part in err["loc"]) or "top level"
        problems.append(f"{loc}: {err['msg']}")
    return "; ".join(problems)


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
            raise CLIError(
                f"Config file at {path} is not valid TOML ({exc}). Fix or delete it.",
                error_type="invalid_config",
                exit_code=2,
            ) from exc
    try:
        cfg = Config.model_validate(data)
    except ValidationError as exc:
        raise CLIError(
            f"Config file at {path} has an unexpected shape "
            f"({_validation_summary(exc)}). Fix or delete it.",
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


def list_profiles() -> dict[str, str | None]:
    """Profile name -> stored backend env, for every profile in config.toml."""
    return {name: prof.env for name, prof in _load().profiles.items()}


def set_active_profile(name: str) -> None:
    """Make ``name`` the default profile for future runs (``assembly config set``).

    Only an existing profile can become active: pointing the default at a name with
    no stored credentials would make every later command fail as "not signed in"
    with no hint why, so the typo is rejected here with the known names listed.
    """
    validate_profile(name)
    cfg = _load()
    if name not in cfg.profiles:
        known = ", ".join(sorted(cfg.profiles)) or "none yet"
        raise CLIError(
            f"No profile named {name!r} (known: {known}).",
            error_type="invalid_profile",
            exit_code=2,
            suggestion=f"Create it first: assembly --profile {name} login",
        )
    cfg.active_profile = name
    _dump(cfg)


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
                "retry (macOS: security delete-generic-password -s assemblyai-cli). "
                "On a headless machine without a keyring, set ASSEMBLYAI_API_KEY instead."
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
    validate_profile(profile)
    _keyring_set(profile, api_key)
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile())
    if cfg.active_profile is None:
        cfg.active_profile = profile
    _dump(cfg)


def _keyring_get(username: str) -> str | None:
    """Read a secret, treating an unusable keyring backend as "nothing stored".

    Headless machines (containers, CI, servers) routinely have no keyring backend at
    all, so keyring raises NoKeyringError on every read. That state must read as "not
    signed in" — ASSEMBLYAI_API_KEY still works there — never as a crash.
    """
    try:
        return keyring.get_password(KEYRING_SERVICE, username)
    except keyring.errors.KeyringError:
        return None


def get_api_key(profile: str) -> str | None:
    return _keyring_get(profile)


def keyring_usable() -> bool:
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


def get_profile_env(profile: str) -> str | None:
    """The backend environment recorded for a profile, if any (e.g. 'sandbox000')."""
    prof = _load().profiles.get(profile)
    return prof.env if prof else None


def set_profile_env(profile: str, env: str) -> None:
    """Bind a backend environment to a profile so its key and hosts stay matched."""
    validate_profile(profile)
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile()).env = env
    _dump(cfg)


def get_profile_email(profile: str) -> str | None:
    """The login email recorded for a profile at browser login, if any."""
    prof = _load().profiles.get(profile)
    return prof.email if prof else None


def set_profile_email(profile: str, email: str) -> None:
    """Persist the login email for a profile (gates internal-environment access)."""
    validate_profile(profile)
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile()).email = email
    _dump(cfg)


def clear_api_key(profile: str) -> None:
    # KeyringError, not just PasswordDeleteError: with no backend at all (headless
    # boxes) delete raises NoKeyringError, and "nothing stored" is already the goal.
    with contextlib.suppress(keyring.errors.KeyringError):
        keyring.delete_password(KEYRING_SERVICE, profile)


SESSION_KEYRING_PREFIX = "session"  # keyring username: f"{prefix}:{profile}"


def _session_username(profile: str) -> str:
    return f"{SESSION_KEYRING_PREFIX}:{profile}"


def set_session(profile: str, *, session_jwt: str, session_token: str, account_id: int) -> None:
    """Persist the browser-login Stytch session (secret) + account id (non-secret).

    AMS self-service endpoints authenticate with this session cookie, not the API
    key. The JWT is short-lived; an expired session surfaces as NotAuthenticated.
    """
    validate_profile(profile)
    _keyring_set(
        _session_username(profile),
        StoredSession(jwt=session_jwt, token=session_token).model_dump_json(),
    )
    cfg = _load()
    cfg.profiles.setdefault(profile, Profile()).account_id = account_id
    _dump(cfg)


def get_session(profile: str) -> dict[str, str] | None:
    """The stored {'jwt', 'token'} for a profile, or None if absent/corrupt."""
    raw = _keyring_get(_session_username(profile))
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
    with contextlib.suppress(keyring.errors.KeyringError):
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
    email: str | None = None,
) -> None:
    """Atomically persist a full browser-login result (API key + env + session).

    The three writes span the keyring and config.toml, so a mid-sequence failure
    (e.g. a locked keychain after the key is already stored) would otherwise leave a
    half-written profile — an API key with no session, which looks signed-in but
    can't reach AMS. On any failure the pre-login snapshot is restored: config.toml
    is rewritten verbatim in one atomic dump, and the two keyring entries are
    restored best-effort.
    """
    validate_profile(profile)
    prior_api_key = _keyring_get(profile)
    prior_session = _keyring_get(_session_username(profile))
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
        # Within the same atomic rollback so the sandbox gate can't read stale identity.
        if email is not None:
            set_profile_email(profile, email)
        done = True
    finally:
        if not done:
            _keyring_restore(profile, prior_api_key)
            _keyring_restore(_session_username(profile), prior_session)
            _dump(prior_cfg)


def has_device_id() -> bool:
    """Whether the anonymous telemetry device id has been minted yet, without
    minting one — lets telemetry detect the true first run for its one-time
    collection disclosure."""
    return _load().device_id is not None


def get_device_id() -> str:
    """A stable anonymous install id for telemetry: a random UUID minted locally on
    first use and persisted in config.toml. Carries nothing derivable from the
    machine or account."""
    cfg = _load()
    if cfg.device_id is None:
        cfg.device_id = str(uuid.uuid4())
        _dump(cfg)
    return cfg.device_id


def get_telemetry_enabled() -> bool | None:
    """The persisted telemetry choice: True/False if the user ran
    `aai telemetry enable/disable`, None if they never chose."""
    return _load().telemetry_enabled


def set_telemetry_enabled(*, enabled: bool) -> None:
    cfg = _load()
    cfg.telemetry_enabled = enabled
    _dump(cfg)


def get_update_cache() -> tuple[float | None, str | None]:
    """The cached (last-check unix ts, latest version seen) for the update notifier."""
    cfg = _load()
    return cfg.update_last_check, cfg.update_latest_version


def set_update_cache(*, last_check: float, latest_version: str | None) -> None:
    """Persist the update-notifier cache. ``latest_version`` is None when the last
    fetch failed — the timestamp is still recorded so we don't re-spawn every run."""
    cfg = _load()
    cfg.update_last_check = last_check
    cfg.update_latest_version = latest_version
    _dump(cfg)


def resolve_api_key(*, profile: str | None = None, api_key_flag: str | None = None) -> str:
    """The API key for SDK/gateway calls: --api-key flag > ASSEMBLYAI_API_KEY > keyring.

    Every resolved key is registered with the verbose-log redactor
    (``debuglog.register_secret``) at this single choke point, so ``-v``/``-vv``
    diagnostics can never print it in clear no matter which library logs it.
    """
    key = _resolve_api_key(profile=profile, api_key_flag=api_key_flag)
    debuglog.register_secret(key)
    return key


def _resolve_api_key(*, profile: str | None, api_key_flag: str | None) -> str:
    # Values are stripped at every tier: a whitespace-only key (e.g. a botched
    # `export ASSEMBLYAI_API_KEY='   '`) must read as "no key" (the clean exit-4
    # not-signed-in path), not get sent as an illegal HTTP header byte string.
    if api_key_flag is not None:
        flag_key = api_key_flag.strip()
        if not flag_key:
            raise CLIError(
                "Empty --api-key provided.",
                error_type="invalid_key",
                exit_code=2,
                suggestion="Pass a non-empty key, e.g. --api-key sk_...",
            )
        return flag_key
    env_key = (env.get(ENV_API_KEY) or "").strip()
    if env_key:
        return env_key
    profile = profile or get_active_profile()
    stored = (get_api_key(profile) or "").strip()
    if stored:
        return stored
    raise NotAuthenticated()


def resolve_api_key_optional(*, profile: str | None = None) -> str | None:
    """The same key chain as ``resolve_api_key`` (env -> keyring), but ``None`` instead
    of raising when no key is configured — for callers that work without one
    (``assembly init`` scaffolding, the onboarding wizard's signed-in check)."""
    try:
        return resolve_api_key(profile=profile)
    except NotAuthenticated:
        return None
