from __future__ import annotations

from assemblyai_cli.render import BaseRenderer


class AgentRenderer(BaseRenderer):
    """Renders Voice Agent events: human transcript lines, or NDJSON for agents.

    Audio payloads are never written; only text/state events are surfaced.
    """

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit({"type": "session.ready"})
        else:
            self._write("Connected — start talking. (Ctrl-C to stop)\n")

    def notice(self, text: str) -> None:
        """Write a human-facing notice line (caller chooses when to suppress in JSON)."""
        self._write(text)

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user.delta", "text": text})
            return
        self._update_line("you: " + text)

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user", "text": text})
            return
        self._finalize_line("you: " + text)

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit({"type": "reply.started"})

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.agent", "text": text, "interrupted": interrupted})
            return
        self._finalize_line()  # close any open "you: …" partial first
        self._write("agent: " + text + "\n")

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "reply.done", "interrupted": interrupted})
