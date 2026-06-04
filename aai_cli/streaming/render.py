from __future__ import annotations

from rich.text import Text

from aai_cli.render import BaseRenderer


class StreamRenderer(BaseRenderer):
    """Renders streaming events in one of three modes.

    - JSON: newline-delimited JSON to stdout (pipe-safe, machine-readable).
    - text: only finalized turn transcripts, one plain line each, to stdout; status
      notices ("Listening…") go to stderr. Lets `aai stream -o text | aai llm "…"`
      pipe clean transcript text downstream.
    - human (default): a live-updating line through Rich.

    Construction and the json/text/human plumbing live in BaseRenderer.
    """

    def begin(self, event: object) -> None:
        # The "Listening…" notice waits for the mic (see listening()); opening the
        # session only emits the protocol event for JSON consumers.
        if self.json_mode:
            self._emit({"type": "begin", "id": getattr(event, "id", None)})

    def listening(self) -> None:
        """Announce capture has started — called once the mic is open and recording."""
        if self.text_mode:
            self._status("Listening… (Ctrl-C to stop)")
        elif not self.json_mode:
            self._line(Text("Listening… (Ctrl-C to stop)", style="aai.muted"))

    def turn(self, event: object) -> None:
        text = getattr(event, "transcript", "") or ""
        end = bool(getattr(event, "end_of_turn", False))
        if self.json_mode:
            self._emit({"type": "turn", "transcript": text, "end_of_turn": end})
        elif self.text_mode:
            if end and text:
                self._write(text + "\n")  # plain finalized line, pipe-friendly
        elif end:
            self._finalize_line(text)
        else:
            self._update_line(text)

    def termination(self, event: object) -> None:
        if self.json_mode:
            self._emit(
                {
                    "type": "termination",
                    "audio_duration_seconds": getattr(event, "audio_duration_seconds", None),
                }
            )

    def stopped(self) -> None:
        if self.text_mode:
            self._status("Stopped.")
        else:
            super().stopped()
