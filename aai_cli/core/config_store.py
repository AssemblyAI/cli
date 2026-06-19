"""The config.toml document schema and its atomic read-modify-write store.

Factored out of ``config`` the same way ``keyring_store`` holds the keyring
access: ``config`` is the auth/profile facade, and this module is the layer
beneath it — the pydantic models that describe ``config.toml`` plus the
parse/cache/atomic-dump machinery that reads and writes the file. Keeping the
two apart means the "every write is a temp-file + atomic ``os.replace``" rule
is structural, and the facade reads as plain accessors over ``load``/``dump``.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import time
import tomllib
from collections.abc import Callable, Generator
from pathlib import Path

import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_cli.core.errors import CLIError


class Profile(BaseModel):
    """A single profile's non-secret settings persisted in config.toml.

    ``extra="allow"`` so unknown keys written by a newer CLI survive a round-trip
    through an older one instead of being silently dropped on the next ``dump``.
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


# Parsed-config cache: path -> (mtime_ns, size, parsed). The several load()
# calls in one CLI invocation (profile, env, key resolution) then don't each
# re-read and re-parse the same unchanged TOML; dump() bumps the mtime, which
# invalidates it naturally. Callers mutate the returned Config (and persist_login
# snapshots one for rollback), so hand out deep copies, never the cached object.
_load_cache: dict[Path, tuple[int, int, Config]] = {}

# Windows has no atomic replace-over-open like POSIX: while dump() swaps the temp
# file in (os.replace), a racing open on the same path transiently fails with
# PermissionError. Since readers are lock-free, both a lock-free reader's open and
# the writer's replace can lose that race, so each retries a few short backoffs to
# ride out the (sub-millisecond) rename window. POSIX replaces atomically and never
# raises here, so this only ever loops on Windows.
_SHARING_RETRIES = 5  # pragma: no mutate -- a ±1 change in the retry budget is equivalent
_SHARING_BACKOFF = 0.02  # pragma: no mutate -- a timing constant; any small value works


def _retry_on_sharing_violation[T](op: Callable[[], T]) -> T:
    """Run a file op, retrying the transient PermissionError Windows raises when an
    open and an os.replace race on the same path (see _SHARING_RETRIES)."""
    for _ in range(_SHARING_RETRIES - 1):
        try:
            return op()
        except PermissionError:
            time.sleep(_SHARING_BACKOFF)
    return op()  # the last attempt's error (a genuine permission problem) propagates


def load() -> Config:
    path = _config_file()
    try:
        stat = path.stat()
    except OSError:
        return Config()
    cached = _load_cache.get(path)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2].model_copy(deep=True)
    raw = _retry_on_sharing_violation(path.read_bytes)
    try:
        data = tomllib.load(io.BytesIO(raw))
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


def dump(cfg: Config) -> None:
    path = _config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and atomically rename over the target, so a crash
    # (or concurrent reader) mid-write can never leave config.toml truncated into
    # invalid TOML that load() would then reject. os.replace is atomic within a dir.
    # exclude_none is required: TOML has no null and tomli_w rejects None values.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".toml.tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            tomli_w.dump(cfg.model_dump(exclude_none=True), fh)
        _retry_on_sharing_violation(lambda: tmp.replace(path))
        # The mtime/size key usually invalidates on its own, but drop the entry
        # explicitly so a same-size rewrite on a coarse-mtime filesystem can't
        # serve the pre-write parse.
        _load_cache.pop(path, None)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


@contextlib.contextmanager
def update() -> Generator[Config]:
    """Run a config.toml read-modify-write: ``load`` -> mutate the yielded config ->
    ``dump``. The dump runs on clean exit; an exception in the block propagates and
    skips it. The atomic os.replace in ``dump`` keeps a reader from ever seeing a torn
    file (writers and readers are otherwise unsynchronized: last write wins)."""
    cfg = load()
    yield cfg
    dump(cfg)
