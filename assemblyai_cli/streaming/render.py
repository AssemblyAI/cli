from __future__ import annotations

import sys
from typing import TextIO

from rich.console import Console
from rich.text import Text

from assemblyai_cli.render import BaseRenderer


class StreamRenderer(BaseRenderer):
    """Renders streaming events in one of three modes.

    - JSON: newline-delimited JSON to stdout (pipe-safe, machine-readable).
    - text: only finalized turn transcripts, one plain line each, to stdout; status
      notices ("Listening…") go to stderr. Lets `aai stream -o text | aai llm "…"`
      pipe clean transcript text downstream.
    - human (default): a live-updating line through Rich.
    """

    def __init__(
        self,
        *,
        json_mode: bool,
        text_mode: bool = False,
        out: TextIO | None = None,
        console: Console | None = None,
        err: TextIO | None = None,
    ) -> None:
        super().__init__(json_mode=json_mode, out=out, console=console)
        self.text_mode = text_mode
        self._err = err if err is not None else sys.stderr

    def _status(self, message: str) -> None:
        """Write a status notice to stderr so it never pollutes piped stdout."""
        print(message, file=self._err, flush=True)

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

    def llm(self, content: str) -> None:
        """Render the LLM Gateway transform of the full transcript (shown last)."""
        if not content:
            return
        if self.json_mode:
            self._emit({"type": "llm", "content": content})
        elif self.text_mode:
            self._write(content + "\n")
        else:
            self._line(Text("\N{ELECTRIC LIGHT BULB} " + content, style="aai.brand"))

    def stopped(self) -> None:
        if self.text_mode:
            self._status("Stopped.")
        else:
            super().stopped()
