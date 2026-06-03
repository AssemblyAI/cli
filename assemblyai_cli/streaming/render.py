from __future__ import annotations

import contextlib
import json
import sys
from typing import TextIO


class StreamRenderer:
    """Renders streaming events: a live-updating line for humans, NDJSON for agents."""

    def __init__(self, *, json_mode: bool, out: TextIO | None = None) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self._line_open = False

    def begin(self, event: object) -> None:
        if self.json_mode:
            self._emit({"type": "begin", "id": getattr(event, "id", None)})
        else:
            self.out.write("Listening… (Ctrl-C to stop)\n")
            self.out.flush()

    def turn(self, event: object) -> None:
        text = getattr(event, "transcript", "") or ""
        end = bool(getattr(event, "end_of_turn", False))
        if self.json_mode:
            self._emit({"type": "turn", "transcript": text, "end_of_turn": end})
            return
        self.out.write("\r\x1b[K" + text)  # clear the line, then write the current turn
        if end:
            self.out.write("\n")
            self._line_open = False
        else:
            self._line_open = True
        self.out.flush()

    def termination(self, event: object) -> None:
        if self.json_mode:
            self._emit(
                {
                    "type": "termination",
                    "audio_duration_seconds": getattr(event, "audio_duration_seconds", None),
                }
            )

    def close(self) -> None:
        """Finalize an in-progress (no-newline) human line, so later output starts clean."""
        if self.json_mode or not self._line_open:
            return
        self._line_open = False
        with contextlib.suppress(Exception):  # the downstream pipe may already be closed
            self.out.write("\n")
            self.out.flush()

    def _emit(self, obj: object) -> None:
        self.out.write(json.dumps(obj) + "\n")
        self.out.flush()
