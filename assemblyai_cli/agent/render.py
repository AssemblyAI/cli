from __future__ import annotations

import json
import sys


class AgentRenderer:
    """Renders Voice Agent events: human transcript lines, or NDJSON for agents.

    Audio payloads are never written; only text/state events are surfaced.
    """

    def __init__(self, *, json_mode: bool, out=None) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self._partial_open = False

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit({"type": "session.ready"})
        else:
            self._write("Connected — start talking. (Ctrl-C to stop)\n")

    def stopped(self) -> None:
        if not self.json_mode:
            self._write("Stopped.\n")

    def notice(self, text: str) -> None:
        """Write a human-facing notice line (no-op semantics in JSON mode are the caller's choice)."""
        self._write(text)

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user.delta", "text": text})
            return
        self._write("\r\x1b[Kyou: " + text)
        self._partial_open = True

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user", "text": text})
            return
        self._write("\r\x1b[Kyou: " + text + "\n")
        self._partial_open = False

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit({"type": "reply.started"})

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.agent", "text": text, "interrupted": interrupted})
            return
        self._finish_partial()
        self._write("agent: " + text + "\n")

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "reply.done", "interrupted": interrupted})

    # --- teardown ----------------------------------------------------------
    def close(self) -> None:
        if self.json_mode:
            return
        self._finish_partial()

    # --- internals ---------------------------------------------------------
    def _finish_partial(self) -> None:
        if self._partial_open:
            self._partial_open = False
            self._write("\n")

    def _emit(self, obj) -> None:
        self._write(json.dumps(obj) + "\n")

    def _write(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except Exception:  # noqa: BLE001 - downstream pipe may be closed
            pass
