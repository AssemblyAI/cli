from __future__ import annotations

import contextlib
import json
import sys
from typing import TextIO


class BaseRenderer:
    """Shared plumbing for the streaming and voice-agent renderers.

    Two output modes: newline-delimited JSON for agents/pipes, or a human view
    that redraws a single in-place line (partial transcript) and finalizes it
    with a newline. Subclasses map domain events onto these primitives.
    """

    def __init__(self, *, json_mode: bool, out: TextIO | None = None) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self._line_open = False

    # --- output primitives -------------------------------------------------
    def _emit(self, obj: object) -> None:
        """Write one NDJSON event."""
        self._write(json.dumps(obj) + "\n")

    def _write(self, text: str) -> None:
        with contextlib.suppress(Exception):  # downstream pipe may be closed
            self.out.write(text)
            self.out.flush()

    def _update_line(self, text: str) -> None:
        """Redraw the current line in place (no trailing newline)."""
        self._write("\r\x1b[K" + text)
        self._line_open = True

    def _finalize_line(self, text: str | None = None) -> None:
        """Commit the current line with a newline; optionally replace its text."""
        if text is not None:
            self._write("\r\x1b[K" + text + "\n")
            self._line_open = False
        elif self._line_open:
            self._write("\n")
            self._line_open = False

    # --- shared lifecycle --------------------------------------------------
    def stopped(self) -> None:
        if not self.json_mode:
            self._write("Stopped.\n")

    def close(self) -> None:
        """Finalize an in-progress human line so later output starts clean."""
        if not self.json_mode:
            self._finalize_line()
