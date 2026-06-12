"""Single-keypress input for hotkey-driven commands (`assembly dictate`).

``TerminalKeys`` switches stdin into cbreak mode for the lifetime of a ``with``
block, so individual keypresses arrive without Enter — while Ctrl-C still raises
KeyboardInterrupt (cbreak keeps ISIG, unlike full raw mode). POSIX-only: there
is no termios on Windows, so entering the context raises a clean CLIError there
instead of an ImportError traceback. Stdlib-only on purpose, mirroring the other
non-rendering layers.
"""

from __future__ import annotations

import os
import select
import sys

from aai_cli.errors import CLIError

# Control characters hotkey-driven commands treat as "end the session".
CTRL_C = "\x03"
CTRL_D = "\x04"
ESC = "\x1b"


def _stdin_fd() -> int:
    """The stdin file descriptor, or -1 when stdin has none (a captured/replaced
    stream in an embedding or test harness) — os.isatty(-1) is False, so that
    case falls into the clean not-a-tty error instead of an fileno traceback."""
    try:
        return sys.stdin.fileno()
    except (ValueError, OSError):  # ValueError covers io.UnsupportedOperation
        return -1


class TerminalKeys:
    """Reads single keypresses from a terminal fd, cbreak-scoped via ``with``.

    The fd is injectable (tests drive it through a pty pair); it defaults to
    the process's stdin.
    """

    def __init__(self, fd: int | None = None) -> None:
        self._fd = fd if fd is not None else _stdin_fd()
        # termios.tcgetattr's attribute list (typeshed's exact shape).
        self._saved: list[int | list[bytes | int]] | None = None

    def __enter__(self) -> TerminalKeys:
        try:
            import termios
            import tty
        except ImportError as exc:
            raise CLIError(
                "Hotkey input is not supported on this platform (no termios).",
                error_type="unsupported_platform",
                exit_code=2,
            ) from exc
        if not os.isatty(self._fd):
            raise CLIError(
                "This command needs an interactive terminal: it waits for hotkey presses on stdin.",
                error_type="not_a_tty",
                exit_code=2,
                suggestion="Run it directly in a terminal, without piping or redirecting stdin.",
            )
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

    def read(self, timeout: float | None) -> str | None:
        """One keypress, or None when ``timeout`` elapses or stdin hits EOF.

        ``timeout=None`` blocks until a key arrives; ``timeout=0`` polls without
        waiting (the in-recording check between audio chunks).
        """
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if not data:
            return None
        return data.decode("utf-8", "replace")
