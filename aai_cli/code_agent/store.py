"""Conversation persistence for `assembly code` (deepagents-code parity).

deepagents-code persists sessions in a SQLite checkpoint store so a conversation can
be resumed. We do the same: a SQLite saver under the CLI's config root, keyed by a
session name (reuse the name to resume; pick a new one to start clean). Falling back to
an in-memory saver gives a single ephemeral session.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

_APP = "assemblyai"


def sessions_db_path() -> Path:
    """Path to the SQLite file holding persisted coding sessions (dir created)."""
    root = Path(platformdirs.user_data_dir(_APP)) / "code-sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root / "sessions.sqlite"


def build_checkpointer(*, persist: bool) -> BaseCheckpointSaver:
    """A SQLite checkpoint saver (resumable) when ``persist``, else in-memory."""
    if not persist:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    # check_same_thread=False: the TUI drives the graph from a worker thread.
    conn = sqlite3.connect(str(sessions_db_path()), check_same_thread=False)
    return SqliteSaver(conn)
