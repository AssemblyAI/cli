"""The cross-process write lock for config.toml's read-modify-write.

config.toml's mutating helpers do ``_load -> mutate -> _dump``. The atomic ``os.replace``
in ``_dump`` keeps a *reader* from ever seeing a torn file, but two writers racing would
still lose an update — both read the same config, and the second dump clobbers the first's
change. These helpers serialize the whole read-modify-write across processes via a sibling
lock file; readers stay lock-free (an older-but-valid parse is fine).

Kept out of ``config.py`` only to keep that module under the file-length gate; it reaches
back into ``config`` (its ``_load``/``_dump``/``config_dir``) lazily, at call time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Generator
from pathlib import Path

import filelock

from aai_cli.core import config, locking


def lock_path() -> Path:
    return config.config_dir() / "config.toml.lock"


def write_lock() -> filelock.FileLock:
    """The shared cross-process write lock guarding config.toml."""
    return locking.file_lock(lock_path())


@contextlib.contextmanager
def locked() -> Generator[None]:
    """Hold the config write lock for the duration of the block."""
    with locking.locked(lock_path()):
        yield


@contextlib.contextmanager
def update(
    load: Callable[[], config.Config], dump: Callable[[config.Config], None]
) -> Generator[config.Config]:
    """Run a load -> mutate -> dump under the write lock so a concurrent writer can't lose
    the update. Yields the loaded config; dumps it on clean exit (an exception in the block
    propagates and skips the dump). ``load``/``dump`` are injected by config.py so this
    module stays clear of its private helpers."""
    with locked():
        cfg = load()
        yield cfg
        dump(cfg)
