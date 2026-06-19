from __future__ import annotations

import re
import uuid

from pydantic import ValidationError

from aai_cli.core import config_store, debuglog, env, keyring_store
from aai_cli.core.config_store import Profile, StoredSession
from aai_cli.core.errors import CLIError, NotAuthenticated

ENV_API_KEY = "ASSEMBLYAI_API_KEY"
DEFAULT_PROFILE = "default"

_PROFILE_RE = re.compile(r"^[A-Za-z0-9_-]+$")


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


def get_active_profile() -> str:
    return config_store.load().active_profile or DEFAULT_PROFILE


def list_profiles() -> dict[str, str | None]:
    """Profile name -> stored backend env, for every profile in config.toml."""
    return {name: prof.env for name, prof in config_store.load().profiles.items()}


def set_active_profile(name: str) -> None:
    """Make ``name`` the default profile for future runs (``assembly config set``).

    Only an existing profile can become active: pointing the default at a name with
    no stored credentials would make every later command fail as "not signed in"
    with no hint why, so the typo is rejected here with the known names listed.
    """
    validate_profile(name)
    with config_store.update() as cfg:
        if name not in cfg.profiles:
            known = ", ".join(sorted(cfg.profiles)) or "none yet"
            raise CLIError(
                f"No profile named {name!r} (known: {known}).",
                error_type="invalid_profile",
                exit_code=2,
                suggestion=f"Create it first: assembly --profile {name} login",
            )
        cfg.active_profile = name


def set_api_key(profile: str, api_key: str) -> None:
    validate_profile(profile)
    keyring_store.set_secret(profile, api_key)
    with config_store.update() as cfg:
        cfg.profiles.setdefault(profile, Profile())
        if cfg.active_profile is None:
            cfg.active_profile = profile


def get_api_key(profile: str) -> str | None:
    return keyring_store.get_secret(profile)


def keyring_usable() -> bool:
    """True when the OS keyring backend can be read (delegates to ``keyring_store``).

    Kept on ``config`` as part of the auth-state facade: ``assembly doctor``/`login`
    call it to tell a user with no key that the *backend* is the problem — and to
    recommend ASSEMBLYAI_API_KEY — rather than pointing at the browser flow, which
    also can't persist on a keyring-less box.
    """
    return keyring_store.usable()


def _profile(profile: str) -> Profile | None:
    """The stored profile record, or None if the profile has none yet."""
    return config_store.load().profiles.get(profile)


def get_profile_env(profile: str) -> str | None:
    """The backend environment recorded for a profile, if any (e.g. 'sandbox000')."""
    prof = _profile(profile)
    return prof.env if prof else None


def set_profile_env(profile: str, env: str) -> None:
    """Bind a backend environment to a profile so its key and hosts stay matched."""
    validate_profile(profile)
    with config_store.update() as cfg:
        cfg.profiles.setdefault(profile, Profile()).env = env


def get_profile_email(profile: str) -> str | None:
    """The login email recorded for a profile at browser login, if any."""
    prof = _profile(profile)
    return prof.email if prof else None


def set_profile_email(profile: str, email: str) -> None:
    """Persist the login email for a profile (gates internal-environment access)."""
    validate_profile(profile)
    with config_store.update() as cfg:
        cfg.profiles.setdefault(profile, Profile()).email = email


def clear_api_key(profile: str) -> None:
    keyring_store.delete_secret(profile)


SESSION_KEYRING_PREFIX = "session"  # keyring username: f"{prefix}:{profile}"


def _session_username(profile: str) -> str:
    return f"{SESSION_KEYRING_PREFIX}:{profile}"


def set_session(profile: str, *, session_jwt: str, session_token: str, account_id: int) -> None:
    """Persist the browser-login Stytch session (secret) + account id (non-secret).

    AMS self-service endpoints authenticate with this session cookie, not the API
    key. The JWT is short-lived; an expired session surfaces as NotAuthenticated.
    """
    validate_profile(profile)
    keyring_store.set_secret(
        _session_username(profile),
        StoredSession(jwt=session_jwt, token=session_token).model_dump_json(),
    )
    with config_store.update() as cfg:
        cfg.profiles.setdefault(profile, Profile()).account_id = account_id


def get_session(profile: str) -> dict[str, str] | None:
    """The stored {'jwt', 'token'} for a profile, or None if absent/corrupt."""
    raw = keyring_store.get_secret(_session_username(profile))
    if not raw:
        return None
    try:
        session = StoredSession.model_validate_json(raw)
    except ValidationError:
        return None
    return {"jwt": session.jwt, "token": session.token}


def get_account_id(profile: str) -> int | None:
    """The AMS account id recorded at login for a profile, if any."""
    prof = _profile(profile)
    return prof.account_id if prof else None


def clear_session(profile: str) -> None:
    keyring_store.delete_secret(_session_username(profile))
    cfg = config_store.load()
    prof = cfg.profiles.get(profile)
    if prof and prof.account_id is not None:
        prof.account_id = None
        config_store.dump(cfg)


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
    # Snapshot the prior state so a mid-sequence failure rolls back cleanly to it.
    prior_api_key = keyring_store.get_secret(profile)
    prior_session = keyring_store.get_secret(_session_username(profile))
    prior_cfg = config_store.load()
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
            keyring_store.restore_secret(profile, prior_api_key)
            keyring_store.restore_secret(_session_username(profile), prior_session)
            config_store.dump(prior_cfg)


def has_device_id() -> bool:
    """Whether the anonymous telemetry device id has been minted yet, without
    minting one — lets telemetry detect the true first run for its one-time
    collection disclosure."""
    return config_store.load().device_id is not None


def get_device_id() -> str:
    """A stable anonymous install id for telemetry: a random UUID minted locally on
    first use and persisted in config.toml. Carries nothing derivable from the
    machine or account."""
    cfg = config_store.load()
    if cfg.device_id is None:
        cfg.device_id = str(uuid.uuid4())
        config_store.dump(cfg)
    return cfg.device_id


def get_telemetry_enabled() -> bool | None:
    """The persisted telemetry choice: True/False if the user ran
    `aai telemetry enable/disable`, None if they never chose."""
    return config_store.load().telemetry_enabled


def set_telemetry_enabled(*, enabled: bool) -> None:
    with config_store.update() as cfg:
        cfg.telemetry_enabled = enabled


def get_update_cache() -> tuple[float | None, str | None]:
    """The cached (last-check unix ts, latest version seen) for the update notifier."""
    cfg = config_store.load()
    return cfg.update_last_check, cfg.update_latest_version


def set_update_cache(*, last_check: float, latest_version: str | None) -> None:
    """Persist the update-notifier cache. ``latest_version`` is None when the last
    fetch failed — the timestamp is still recorded so we don't re-spawn every run."""
    with config_store.update() as cfg:
        cfg.update_last_check = last_check
        cfg.update_latest_version = latest_version


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
