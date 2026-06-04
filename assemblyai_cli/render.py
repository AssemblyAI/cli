from __future__ import annotations

import json
import sys
from typing import TextIO

from rich.console import Console
from rich.live import Live
from rich.text import Text

from assemblyai_cli import theme


class BaseRenderer:
    """Shared plumbing for the streaming and voice-agent renderers.

    Two output modes. JSON mode writes newline-delimited JSON straight to the
    stream (pipe-safe). Human mode renders through Rich: an in-progress line is
    shown with `rich.live.Live` (which redraws and clears multi-row wraps
    cleanly), and finalized lines are printed above it as permanent scrollback.
    """

    def __init__(
        self,
        *,
        json_mode: bool,
        out: TextIO | None = None,
        console: Console | None = None,
        text_mode: bool = False,
        err: TextIO | None = None,
    ) -> None:
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        # text mode emits plain transcript lines to stdout and status notices to
        # stderr, so piping never mixes the two; err defaults to real stderr.
        self.text_mode = text_mode
        self._err = err if err is not None else sys.stderr
        self._console = console
        self._live: Live | None = None

    def _status(self, message: str) -> None:
        """Write a status notice to stderr so it never pollutes piped stdout."""
        print(message, file=self._err, flush=True)

    # --- JSON output (plain text; preserves BrokenPipe for `| head`) -------
    def _emit(self, obj: object) -> None:
        """Write one NDJSON event."""
        self._write(json.dumps(obj) + "\n")

    def _write(self, text: str) -> None:
        try:
            self.out.write(text)
            self.out.flush()
        except BrokenPipeError:
            # Consumer (e.g. `| head`) went away — let the command stop cleanly.
            raise
        except Exception:  # noqa: BLE001, S110 - other downstream write errors are non-fatal
            pass

    # --- human output (Rich) ----------------------------------------------
    def _console_obj(self) -> Console:
        if self._console is None:
            self._console = theme.make_console(file=self.out)
        return self._console

    def _live_obj(self) -> Live:
        if self._live is None:
            # redirect_stdout/stderr stay off: Live must not hijack the process
            # streams that the JSON path and threaded callbacks also write to.
            self._live = Live(
                console=self._console_obj(),
                auto_refresh=False,
                transient=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        return self._live

    def _commit_live(self) -> None:
        """Stop the live region, leaving its last frame as a permanent line."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    @staticmethod
    def _as_text(text: str | Text) -> Text:
        return text if isinstance(text, Text) else Text(text)

    def _update_line(self, text: str | Text) -> None:
        """Redraw the in-progress line in place (Rich clears any prior wrap)."""
        self._live_obj().update(self._as_text(text), refresh=True)

    def _finalize_line(self, text: str | Text | None = None) -> None:
        """Commit the in-progress line (optionally replacing its text) as permanent."""
        if self._live is not None:
            if text is not None:
                self._live.update(self._as_text(text), refresh=True)
            self._commit_live()
        elif text is not None:
            self._console_obj().print(self._as_text(text))

    def _line(self, text: str | Text) -> None:
        """Print a standalone permanent line, committing any open partial first."""
        self._commit_live()
        self._console_obj().print(self._as_text(text))

    # --- shared lifecycle --------------------------------------------------
    def stopped(self) -> None:
        if not self.json_mode:
            self._line(Text("Stopped.", style="aai.muted"))

    def close(self) -> None:
        """Commit any in-progress line so later output starts clean."""
        if not self.json_mode:
            self._commit_live()
