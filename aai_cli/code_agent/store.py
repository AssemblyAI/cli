"""Conversation persistence for `assembly code` (deepagents-code parity).

deepagents-code persists sessions in a SQLite checkpoint store so a conversation can
be resumed. We do the same: a SQLite saver under the CLI's config root, keyed by a
session name (reuse the name to resume; pick a new one to start clean). Falling back to
an in-memory saver gives a single ephemeral session.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import platformdirs

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

_APP = "assemblyai"

# Length of a generated session id — short enough to read off the splash and retype as
# ``--session <id>`` to resume, with ample uniqueness for one user's sessions.
_SESSION_ID_LEN = 12


def new_session_id() -> str:
    """A fresh, unique session id so each run starts a clean conversation by default.

    `assembly code` no longer reuses a fixed ``"default"`` thread (which silently resumed the
    previous conversation); each run gets its own id unless ``--session NAME`` names one to
    resume. Shown on the splash as ``Thread: <id>`` so it can be resumed later.
    """
    return uuid.uuid4().hex[:_SESSION_ID_LEN]


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
