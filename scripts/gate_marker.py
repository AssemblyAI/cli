#!/usr/bin/env python3
"""Record / verify that ``scripts/check.sh`` passed for the *current* working tree.

``check.sh`` calls ``record`` on success, writing a signature of the working tree
into the repo's git dir. The pre-commit gate hook
(``.claude/hooks/require-gate-before-commit.sh``) calls ``check``: it recomputes the
signature and blocks ``git commit`` unless it matches — so any edit made *after* the
gate passed invalidates the marker and re-requires a green run.

stdlib-only on purpose (runs from a hook and from check.sh, before/without ``uv``).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Protocol


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


MARKER_NAME = "aai-gate-pass"
# Hash full content for normal source files; for anything larger (committed audio
# fixtures, build artifacts) fall back to size+mtime so a commit check stays fast.
_CONTENT_HASH_MAX_BYTES = 2 * 1024 * 1024
# `git status --porcelain=v1 -z` records are "XY<space><path>" — 3-byte prefix.
_STATUS_PREFIX_LEN = 3
# argv when invoked as `gate_marker.py <action>`.
_ARGC_WITH_ACTION = 2


def _git(*args: str) -> bytes:
    # S603/S607 are ignored project-wide (the CLI shells out to controlled tools);
    # args here are all literals, never user input.
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def _marker_path() -> Path:
    git_dir = _git("rev-parse", "--git-dir").decode().strip()
    return Path(git_dir) / MARKER_NAME


def _head() -> bytes:
    try:
        return _git("rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError:
        return b"NO_HEAD"


def _changed_paths() -> list[str]:
    """Paths git reports as changed/untracked, independent of staging state.

    ``--no-renames`` keeps one path per record (rename -> delete + add) so each
    NUL-terminated record is ``XY<space><path>``. The path set is identical whether
    a change is staged or not, which makes the signature stable across ``git add``.
    """
    out = _git("status", "--porcelain=v1", "--no-renames", "-z")
    paths: list[str] = []
    for record in out.split(b"\x00"):
        if len(record) <= _STATUS_PREFIX_LEN:
            continue
        paths.append(record[_STATUS_PREFIX_LEN:].decode("utf-8", "surrogateescape"))
    return sorted(paths)


def _update_file(path: Path, digest: _Digest) -> None:
    """Fingerprint one regular file: full content when small, else size+mtime."""
    try:
        stat = path.stat()
    except OSError:
        digest.update(b"\x00UNREADABLE")
        return
    if stat.st_size <= _CONTENT_HASH_MAX_BYTES:
        digest.update(b"\x00C")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"UNREADABLE")
    else:
        digest.update(f"\x00S{stat.st_size}:{stat.st_mtime_ns}".encode())


def _update_for_path(rel: str, digest: _Digest) -> None:
    digest.update(rel.encode("utf-8", "surrogateescape"))
    path = Path(rel)
    if not path.exists():
        digest.update(b"\x00DELETED")
    elif path.is_dir():
        # git collapses a fully-untracked directory into one `dir/` entry. Such dirs
        # are scratch (e.g. tmp/), not what the gate validates, so fingerprint their
        # files by name+stat only — cheap, deterministic, and enough to notice drift.
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child).encode("utf-8", "surrogateescape"))
            try:
                cst = child.stat()
                digest.update(f"\x00S{cst.st_size}:{cst.st_mtime_ns}".encode())
            except OSError:
                digest.update(b"\x00UNREADABLE")
    else:
        _update_file(path, digest)


def _signature() -> str:
    digest = hashlib.sha256()
    digest.update(_head())
    digest.update(b"\x00")
    for rel in _changed_paths():
        _update_for_path(rel, digest)
    return digest.hexdigest()


def _cmd_record() -> int:
    _marker_path().write_text(_signature(), encoding="utf-8")
    return 0


def _cmd_check() -> int:
    marker = _marker_path()
    if not marker.exists():
        return 1
    return 0 if marker.read_text(encoding="utf-8").strip() == _signature() else 1


def main(argv: list[str]) -> int:
    if len(argv) != _ARGC_WITH_ACTION or argv[1] not in {"record", "check"}:
        sys.stderr.write("usage: gate_marker.py {record|check}\n")
        return 2
    return _cmd_record() if argv[1] == "record" else _cmd_check()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
