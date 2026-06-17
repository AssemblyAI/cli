"""Cross-process advisory file locking, backed by ``filelock``.

Used to serialize a read-modify-write of a shared on-disk file (``config.toml``) across
concurrent ``assembly`` processes, so two of them can't lose each other's updates
(last-writer-wins). The atomic-rename write keeps a *reader* from ever seeing a torn
file; this lock is what keeps two *writers* from clobbering each other.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path

import filelock

# One cached lock instance per lock-file path. filelock already serializes threads within
# this process (and other processes via the lock file), and reusing a single instance per
# path makes nested acquisitions reentrant — distinct instances on one path deadlock, which
# is what a re-entrant caller (e.g. a snapshot+rollback that calls smaller writers) needs.
_lock_cache: dict[str, filelock.FileLock] = {}


def file_lock(path: Path) -> filelock.FileLock:
    """The cached cross-process lock for ``path`` (created on first use)."""
    key = str(path)
    lock = _lock_cache.get(key)
    if lock is None:
        lock = filelock.FileLock(path)
        _lock_cache[key] = lock
    return lock


@contextlib.contextmanager
def locked(path: Path) -> Generator[None]:
    """Hold the cross-process lock at ``path`` for the duration of the block.

    Creates the lock file's parent directory first — filelock won't, and the very first
    write on a fresh machine targets a dir that may not exist yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        yield
