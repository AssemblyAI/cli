"""Signal plumbing for the long-running interactive commands.

`stream` (and the other realtime commands) treat Ctrl-C as a clean "user stopped"
signal: SIGINT arrives as a ``KeyboardInterrupt`` that the streaming lifecycle catches
to flush its closing state and exit 130 (the cancel code). This module lets an
*external* stop reach that same path — see ``terminate_as_interrupt``.

`dictate` wants the opposite of cancel: an external SIGTERM means "I'm done, now
transcribe" (a clean exit 0), so it uses ``stop_on_terminate`` instead — same
SIGTERM, latched into a poll-able flag rather than re-raised as a cancel.
"""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Callable, Generator
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


class _StopFlag:
    """A poll-able 'SIGTERM has been delivered' latch (see ``stop_on_terminate``)."""

    def __init__(self) -> None:
        self.stopped = False

    def __call__(self) -> bool:
        return self.stopped


@contextlib.contextmanager
def stop_on_terminate() -> Generator[Callable[[], bool]]:
    """Latch SIGTERM into a poll-able 'stop now' predicate for the block.

    Unlike ``terminate_as_interrupt`` (which routes SIGTERM into the cancel path),
    this is for `assembly dictate`'s headless capture: a controller — a Hammerspoon
    hotkey, a wrapper's ``kill -TERM`` — launches `assembly dictate`, which starts
    recording immediately, then sends SIGTERM to mean "I'm done dictating". The
    capture loop sees the flag flip, stops, and the utterance is transcribed (a clean
    exit 0). SIGINT (Ctrl-C) stays the cancel that aborts without transcribing.

    Yields a zero-argument predicate the capture loop polls between mic chunks; it
    flips to True once SIGTERM is delivered. No-op off the main thread (where
    ``signal.signal`` refuses to install a handler), yielding a predicate that stays
    False; the previous handler is always restored on exit.
    """
    flag = _StopFlag()
    if threading.current_thread() is not threading.main_thread():
        yield flag
        return

    def _request_stop(signum: int, frame: FrameType | None) -> None:
        del signum, frame  # handler signature is fixed; the values are unused
        flag.stopped = True

    previous = signal.signal(signal.SIGTERM, _request_stop)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGTERM, previous)
