from __future__ import annotations

from typing import Any

from rich.text import Text

from assemblyai_cli.render import BaseRenderer


def _labeled(label: str, body: str) -> Text:
    """A line whose `label` prefix is brand-accented and whose body is default."""
    return Text.assemble((label, "aai.label"), body)


class AgentRenderer(BaseRenderer):
    """Renders Voice Agent events: human transcript lines, or NDJSON for agents.

    Audio payloads are never written; only text/state events are surfaced.
    """

    def __init__(self, *, mic_input: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # File-driven runs have no mic, so they skip the "start talking" prompt.
        self.mic_input = mic_input

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit({"type": "session.ready"})
        elif self.mic_input:
            self._line(Text("Connected — start talking. (Ctrl-C to stop)", style="aai.muted"))

    def notice(self, text: str) -> None:
        """Print a human-facing notice (caller chooses when to suppress in JSON)."""
        self._line(text.rstrip("\n"))

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user.delta", "text": text})
            return
        self._update_line(_labeled("you: ", text))

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.user", "text": text})
            return
        self._finalize_line(_labeled("you: ", text))

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit({"type": "reply.started"})

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "transcript.agent", "text": text, "interrupted": interrupted})
            return
        self._line(_labeled("agent: ", text))  # commits any open "you: …" partial first

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit({"type": "reply.done", "interrupted": interrupted})
