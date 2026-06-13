"""The config.toml persistence layer: the document schema plus atomic, cached
read/write. No secrets and no keyring live here — `config.py` is the credential and
profile API that calls this store. Kept Rich-free (like `config.py`) so the lowest
library layers never depend on rendering.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import tomllib
from pathlib import Path

import platformdirs
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aai_cli.errors import CLIError


class Profile(BaseModel):
    """A single profile's non-secret settings persisted in config.toml.

    ``extra="allow"`` so unknown keys written by a newer CLI survive a round-trip
    through an older one instead of being silently dropped on the next ``dump``.
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
    """The platformdirs config directory for the CLI (where ``config.toml`` lives)."""
    return Path(platformdirs.user_config_dir("assemblyai"))


def config_file() -> Path:
    """The path to config.toml inside ``config_dir``."""
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


def load() -> Config:
    """Parse config.toml into a `Config` (a fresh deep copy), serving the mtime/size
    cache on a hit. A malformed or unexpectedly-shaped file becomes a clean CLIError."""
    path = config_file()
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


def dump(cfg: Config) -> None:
    """Persist ``cfg`` to config.toml atomically (temp file + rename) and invalidate
    the load cache, so a crash mid-write can never leave a truncated, invalid file."""
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and atomically rename over the target, so a crash
    # (or concurrent reader) mid-write can never leave config.toml truncated into
    # invalid TOML that load would then reject. os.replace is atomic within a dir.
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
