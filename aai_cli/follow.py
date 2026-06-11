from __future__ import annotations

from rich.live import Live
from rich.markup import escape
from rich.panel import Panel

from aai_cli import output


class FollowRenderer:
    """Render a live transcript transform that refreshes on every turn.

    On a terminal, the latest answer is redrawn in place inside a Rich panel so a
    human watches one evolving summary. When piped or run by an agent (json_mode),
    each refresh is emitted as one NDJSON object so it stays machine-readable.

    Shared by `assembly llm --follow` (turns piped on stdin) and `assembly stream --llm`
    (turns produced live by the streaming session).
    """

    def __init__(self, *, json_mode: bool) -> None:
        self.json_mode = json_mode
        self._live: Live | None = None
        self._last: Panel | None = None

    def __enter__(self) -> FollowRenderer:
        if not self.json_mode:
            # screen=True draws into the terminal's alternate buffer (like less/htop).
            # In the `assembly stream -o text | assembly llm -f` pipeline two processes share one
            # TTY: stream writes status to stderr and the Ctrl-C "^C" echoes into our
            # region, desyncing Rich's relative-cursor teardown and duplicating the top
            # border. The alt buffer is isolated and restored verbatim on exit, so that
            # noise is discarded; we reprint the final panel to the normal screen below.
            self._live = Live(console=output.console, auto_refresh=False, screen=True)
            self._live.start()
        return self

    def __call__(self, answer: str, turns: int) -> None:
        if self.json_mode:
            output.emit_ndjson({"turns": turns, "output": answer})
        elif self._live is not None:
            title = f"scribe · {turns} turn{'s' if turns != 1 else ''}"
            self._last = Panel(escape(answer or "…"), title=title, border_style="aai.brand")
            self._live.update(self._last, refresh=True)

    def __exit__(self, *exc: object) -> None:
        if self._live is not None:
            self._live.stop()  # leaves the alt buffer, restoring the normal screen
            self._live = None
            if self._last is not None:
                output.console.print(self._last)  # leave the final summary as scrollback
