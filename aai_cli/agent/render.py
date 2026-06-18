from __future__ import annotations

from typing import Any

from rich.text import Text

from aai_cli.agent import events
from aai_cli.ui.render import BaseRenderer


def _labeled(label: str, body: str, *, style: str = "aai.label") -> Text:
    """A transcript line tinted entirely in `style` — both the `label` prefix and the body.

    Coloring the whole turn (not just the prefix) is what makes a "you:" line and an
    "agent:" line read as different colors top to bottom; the two styles resolve to
    contrasting hues (see ``theme.aai.you`` / ``theme.aai.agent``).
    """
    return Text(f"{label}{body}", style=style)


class AgentRenderer(BaseRenderer):
    """Renders Voice Agent events in one of three modes.

    - JSON: NDJSON events to stdout. - text: plain ``you:``/``agent:`` transcript
    lines to stdout with status on stderr (so ``assembly agent -o text | assembly llm "…"``
    pipes the conversation). - human (default): live Rich transcript.

    Audio payloads are never written; only text/state events are surfaced.
    """

    def __init__(self, *, mic_input: bool = True, **kwargs: Any) -> None:
        # text_mode/err/json_mode/out/console are handled by BaseRenderer.
        super().__init__(**kwargs)
        # File-driven runs have no mic, so they skip the "start talking" prompt.
        self.mic_input = mic_input

    def _emit_event(self, event: events.Event) -> None:
        """Serialize one typed agent event to the NDJSON stream."""
        self._emit(event.model_dump())

    # --- lifecycle ---------------------------------------------------------
    def connected(self) -> None:
        if self.json_mode:
            self._emit_event(events.SessionReady())
        elif not self.mic_input:
            return
        elif self.text_mode:
            self._status("Connected — start talking. (Ctrl-C to stop)")
        else:
            self._line(Text("Connected — start talking. (Ctrl-C to stop)", style="aai.muted"))

    def notice(self, text: str) -> None:
        """Print a human-facing notice: suppressed in JSON, to stderr otherwise.

        Stderr in *every* non-JSON mode (not just ``-o text``): the default human
        mode is also piped sometimes (``assembly agent | head``), and a notice on stdout
        would be consumed as transcript data there.
        """
        if self.json_mode:
            return
        self._status(text.rstrip("\n"))

    # --- user --------------------------------------------------------------
    def user_partial(self, text: str) -> None:
        if self.json_mode:
            self._emit_event(events.UserDelta(text=text))
        elif not self.text_mode:  # partials are noise for piped text
            self._update_line(_labeled("you: ", text, style="aai.you"))

    def user_final(self, text: str) -> None:
        if self.json_mode:
            self._emit_event(events.UserFinal(text=text))
        elif self.text_mode:
            self._write(f"you: {text}\n")
        else:
            self._finalize_line(_labeled("you: ", text, style="aai.you"))

    # --- agent -------------------------------------------------------------
    def reply_started(self) -> None:
        if self.json_mode:
            self._emit_event(events.ReplyStarted())

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit_event(events.AgentTranscript(text=text, interrupted=interrupted))
        elif self.text_mode:
            self._write(f"agent: {text}\n")
        else:
            # commits any open "you: …" partial first
            self._line(_labeled("agent: ", text, style="aai.agent"))

    def reply_done(self, *, interrupted: bool) -> None:
        if self.json_mode:
            self._emit_event(events.ReplyDone(interrupted=interrupted))
