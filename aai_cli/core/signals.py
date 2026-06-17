"""Signal plumbing for the long-running interactive commands.

`stream` (and the other realtime commands) treat Ctrl-C as a clean "user stopped"
signal: SIGINT arrives as a ``KeyboardInterrupt`` that the streaming lifecycle catches
to flush its closing state and exit 130 (the cancel code). This module lets an
*external* stop reach that same path — see ``terminate_as_interrupt``.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Generator
from types import FrameType


@contextlib.contextmanager
def terminate_as_interrupt() -> Generator[None]:
    """Map SIGTERM onto ``KeyboardInterrupt`` for the duration of the block.

    A supervisor that stops the command from outside the terminal — a Hammerspoon
    hotkey, a service manager, a wrapper script's ``kill`` — sends SIGTERM, whose
    default action aborts the process before the graceful flush runs. Re-raising it
    as ``KeyboardInterrupt`` routes SIGTERM through the same clean-stop path as
    Ctrl-C, so a controller can stop a recording with a plain SIGTERM and never has
    to deliver SIGINT (which, under a ``just``/shell wrapper, means signalling the
    whole process group).

    No-op off the main thread, where ``signal.signal`` refuses to install a handler;
    the previous handler is always restored on exit.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise_interrupt(signum: int, frame: FrameType | None) -> None:
        del signum, frame  # handler signature is fixed; the values are unused
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
