"""Unit tests for aai_cli.core.signals.terminate_as_interrupt."""

from __future__ import annotations

import signal
import threading

import pytest

from aai_cli.core import signals


def test_terminate_as_interrupt_installs_and_restores_handler():
    before = signal.getsignal(signal.SIGTERM)
    with signals.terminate_as_interrupt():
        handler = signal.getsignal(signal.SIGTERM)
        # A new handler is installed for the block...
        assert handler is not before
        assert callable(handler)
        # ...and it turns a SIGTERM into the clean-stop KeyboardInterrupt.
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    # The previous handler is restored on exit.
    assert signal.getsignal(signal.SIGTERM) is before


def test_terminate_as_interrupt_is_noop_off_main_thread():
    before = signal.getsignal(signal.SIGTERM)
    observed: dict[str, object] = {}

    def worker() -> None:
        with signals.terminate_as_interrupt():
            # Off the main thread no handler may be installed, so the disposition
            # is untouched and the block still runs to completion.
            observed["handler"] = signal.getsignal(signal.SIGTERM)
        observed["ran"] = True

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert observed["ran"] is True
    assert observed["handler"] is before
    assert signal.getsignal(signal.SIGTERM) is before
