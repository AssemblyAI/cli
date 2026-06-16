"""Assemble auto-named output paths for `assembly stream --save-dir`.

``--save-dir DIR`` turns filename assembly over to the CLI: the transcript and the
matching WAV land under ``DIR/YYYY-MM-DD/`` with a timestamped, slugged stem
(``YYYY-MM-DD-HHMMSS[-slug]``). That is the block a wrapper script otherwise hand-rolls
with ``date(1)`` plus a ``tr``/``sed`` slug — passing ``--name "<title>"`` feeds e.g. a
calendar event title straight in, so the caller never slugs anything itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aai_cli.core.errors import CLIError

# Each run of non-alphanumeric characters collapses to a single hyphen.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Cap the slug so a long title can't push the filename past a filesystem name limit.
_MAX_SLUG_LEN = 80


def slugify(title: str) -> str:
    """Fold ``title`` to a filename-safe slug.

    Lowercased, with every run of non-alphanumeric characters replaced by a single
    hyphen, capped at 80 characters, and no leading/trailing hyphen. Returns "" when
    the title has no usable characters (e.g. "!!!"), so the caller drops the slug
    suffix rather than emitting a bare-hyphen filename.
    """
    slug = _NON_ALNUM.sub("-", title.lower()).strip("-")
    return slug[:_MAX_SLUG_LEN].strip("-")


def _stem(now: datetime, name: str | None) -> str:
    """The filename stem: a sortable timestamp, plus the slugged name when one survives."""
    stamp = now.strftime("%Y-%m-%d-%H%M%S")
    slug = slugify(name) if name else ""
    return f"{stamp}-{slug}" if slug else stamp


# The sidecar's extension, mirroring batch transcribe's ``<file>.aai.json`` so a
# browse/list UI can recognize a stream recording's metadata file the same way.
SIDECAR_SUFFIX = ".aai.json"


@dataclass(frozen=True)
class SavePaths:
    """The auto-assembled output paths for one recording, all sharing a stem.

    The transcript ``.txt`` and matching audio ``.wav`` plus the optional ``--llm``
    note ``.md`` and the ``.aai.json`` metadata sidecar — each is derived from the
    same ``directory``/``stem`` so they land together and stay matchable by name.
    """

    directory: Path
    stem: str

    @property
    def transcript(self) -> Path:
        return self.directory / f"{self.stem}.txt"

    @property
    def audio(self) -> Path:
        return self.directory / f"{self.stem}.wav"

    @property
    def note(self) -> Path:
        return self.directory / f"{self.stem}.md"

    @property
    def sidecar(self) -> Path:
        return self.directory / f"{self.stem}{SIDECAR_SUFFIX}"


def resolve(save_dir: Path, name: str | None, *, now: datetime) -> SavePaths:
    """Build the ``DIR/YYYY-MM-DD/<stem>`` paths for ``now`` and ``--name``.

    The date bucket and the stem both carry the date so a transcript stays
    self-describing if it is later moved out of its bucket. Path assembly only —
    creating the directory is the caller's job (see ``ensure_dir``).
    """
    bucket = save_dir / now.strftime("%Y-%m-%d")
    return SavePaths(directory=bucket, stem=_stem(now, name))


def channel_audio(audio: Path, channel: str) -> Path:
    """Insert a per-channel suffix into an auto-named WAV path.

    ``--system-audio`` records two parallel streams that can't share one WAV, so each
    channel ("you", "system") gets its own file beside the shared transcript: the base
    ``DIR/.../<stem>.wav`` becomes ``DIR/.../<stem>-<channel>.wav``.
    """
    return audio.with_name(f"{audio.stem}-{channel}{audio.suffix}")


def ensure_dir(path: Path) -> None:
    """Create ``path`` (and parents) for the auto-named files, as a clean CLIError on failure."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CLIError(
            f"Cannot create {path}: {exc}",
            error_type="save_dir_path",
            exit_code=2,
            suggestion="Pass --save-dir under a writable location.",
        ) from exc
