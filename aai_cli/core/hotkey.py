"""Single-keypress input for hotkey-driven commands (`assembly dictate`).

``TerminalKeys`` reads individual keypresses — without waiting for Enter — for the
lifetime of a ``with`` block, while Ctrl-C still ends the program. One interface,
two backends:

- POSIX puts stdin into cbreak mode (termios/tty) and waits with ``select``; cbreak
  keeps ISIG, so Ctrl-C raises KeyboardInterrupt instead of arriving as a byte.
- Windows reads the console through stdlib ``msvcrt`` (``kbhit``/``getwch``), which is
  already character-at-a-time, so there is no mode to enter or restore.

A platform that is neither (no termios and not Windows) raises a clean CLIError rather
than an ImportError traceback. Stdlib-only on purpose, mirroring the other
non-rendering layers.
"""

from __future__ import annotations

import importlib
import os
import select
import sys
import time
from collections.abc import Callable

from aai_cli.core.errors import CLIError

# Control characters hotkey-driven commands treat as "end the session".
CTRL_C = "\x03"
CTRL_D = "\x04"
ESC = "\x1b"

# How long the Windows key poll naps between kbhit() checks (msvcrt has no select()):
# short enough to feel instant at the dictate prompt, long enough not to spin a core.
_WINDOWS_POLL_INTERVAL = 0.01


def _stdin_fd() -> int:
    """The stdin file descriptor, or -1 when stdin has none (a captured/replaced
    stream in an embedding or test harness) — os.isatty(-1) is False, so that
    case falls into the clean not-a-tty error instead of an fileno traceback."""
    try:
        return sys.stdin.fileno()
    except (ValueError, OSError):  # ValueError covers io.UnsupportedOperation
        return -1


def _on_windows() -> bool:
    """True on Windows, where key input goes through msvcrt instead of termios. A
    function (not a constant) so tests can drive the Windows backend on a POSIX host."""
    return sys.platform == "win32"


class TerminalKeys:
    """Reads single keypresses from a terminal, scoped via ``with``.

    The fd is injectable (POSIX tests drive it through a pty pair) and defaults to the
    process's stdin; the Windows backend reads the console directly and ignores it.
    """

    def __init__(self, fd: int | None = None) -> None:
        self._fd = fd if fd is not None else _stdin_fd()
        # termios.tcgetattr's attribute list (typeshed's exact shape); stays None on
        # Windows, where there is no saved terminal state to restore.
        self._saved: list[int | list[bytes | int]] | None = None
        self._windows = _on_windows()

    def __enter__(self) -> TerminalKeys:
        if not os.isatty(self._fd):
            raise CLIError(
                "This command needs an interactive terminal: it waits for hotkey presses on stdin.",
                error_type="not_a_tty",
                exit_code=2,
                suggestion="Run it directly in a terminal, without piping or redirecting stdin.",
            )
        if self._windows:
            # The Windows console is already character-at-a-time; there is no cbreak mode
            # to enter or restore, so read() goes straight through msvcrt.
            return self
        try:
            import termios
            import tty
        except ImportError as exc:
            raise CLIError(
                "Hotkey input is not supported on this platform (no termios).",
                error_type="unsupported_platform",
                exit_code=2,
            ) from exc
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
        if self._windows:
            return self._read_windows(timeout)
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if not data:
            return None
        return data.decode("utf-8", "replace")

    def _read_windows(self, timeout: float | None) -> str | None:
        """Windows key read: getwch() blocks (timeout=None) or kbhit() polls to a deadline."""
        msvcrt = importlib.import_module("msvcrt")
        # Bind the win32-only members to typed locals: typeshed hides them off-Windows,
        # so the type-checkers reject `msvcrt.getwch` directly but accept these.
        kbhit: Callable[[], bool] = msvcrt.kbhit
        getwch: Callable[[], str] = msvcrt.getwch
        if timeout is None:
            return getwch()
        deadline = time.monotonic() + timeout  # pragma: no mutate
        while not kbhit():
            if time.monotonic() >= deadline:  # pragma: no mutate
                return None
            time.sleep(_WINDOWS_POLL_INTERVAL)  # pragma: no mutate
        return getwch()
