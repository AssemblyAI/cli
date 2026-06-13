"""TerminalKeys: cbreak scoping, single-key reads, and the clean failure modes.

The terminal tests drive a real pty pair (os.openpty), so termios behavior is
exercised for real without touching the test runner's stdin.
"""

import os
import sys

import pytest

from aai_cli.core.errors import CLIError
from aai_cli.core.hotkey import TerminalKeys, _stdin_fd

# termios and os.openpty are POSIX-only, so the whole module is skipped on Windows
# (where TerminalKeys raises a clean CLIError rather than running). importorskip keeps
# this out of the skip/xfail escape-hatch count the Linux gate tracks.
termios = pytest.importorskip("termios")


@pytest.fixture
def pty_pair():
    master, slave = os.openpty()
    yield master, slave
    os.close(master)
    os.close(slave)


def test_reads_single_keypresses_without_enter(pty_pair):
    master, slave = pty_pair
    with TerminalKeys(fd=slave) as keys:
        os.write(master, b"ab")
        # One keypress per read, even when several are queued.
        assert keys.read(5.0) == "a"
        assert keys.read(5.0) == "b"


def test_poll_returns_none_when_no_key_is_pending(pty_pair):
    _, slave = pty_pair
    with TerminalKeys(fd=slave) as keys:
        assert keys.read(0) is None


def test_cbreak_is_scoped_to_the_context(pty_pair):
    _, slave = pty_pair
    lflag_index = 3
    assert termios.tcgetattr(slave)[lflag_index] & termios.ICANON
    with TerminalKeys(fd=slave):
        inside = termios.tcgetattr(slave)[lflag_index]
        assert not inside & termios.ICANON  # keys arrive without Enter
        assert inside & termios.ISIG  # but Ctrl-C still raises KeyboardInterrupt
    assert termios.tcgetattr(slave)[lflag_index] & termios.ICANON  # restored


def test_exit_without_enter_restores_nothing(pty_pair):
    # __exit__ is a no-op when the cbreak switch never happened (or already ran):
    # exiting twice must not call tcsetattr with stale state.
    _, slave = pty_pair
    keys = TerminalKeys(fd=slave)
    keys.__exit__(None, None, None)  # never entered: nothing to restore
    with keys:
        pass
    keys.__exit__(None, None, None)  # second exit after restore: still a no-op


def test_non_tty_fd_is_a_clean_usage_error(tmp_path):
    with (tmp_path / "plain-file").open("w") as f:
        with pytest.raises(CLIError) as exc:
            with TerminalKeys(fd=f.fileno()):
                pass
    assert exc.value.exit_code == 2
    assert exc.value.error_type == "not_a_tty"
    assert "interactive terminal" in exc.value.message


def test_platform_without_termios_is_a_clean_error(pty_pair, monkeypatch):
    # Windows has no termios; None in sys.modules makes the import raise.
    _, slave = pty_pair
    monkeypatch.setitem(sys.modules, "termios", None)
    with pytest.raises(CLIError) as exc:
        with TerminalKeys(fd=slave):
            pass
    assert exc.value.exit_code == 2
    assert exc.value.error_type == "unsupported_platform"


def test_read_returns_none_at_eof():
    # A pipe stands in for a hung-up terminal: select reports readable, the
    # read yields no bytes. (read() itself doesn't require a tty; only the
    # cbreak context does.)
    read_end, write_end = os.pipe()
    try:
        os.write(write_end, b"z")
        os.close(write_end)
        keys = TerminalKeys(fd=read_end)
        assert keys.read(0) == "z"  # drains the last byte
        assert keys.read(0) is None  # then EOF
    finally:
        os.close(read_end)


def test_stdin_fd_defaults_to_real_stdin_or_minus_one(monkeypatch):
    class NoFileno:
        def fileno(self):
            raise OSError("no underlying file")

    monkeypatch.setattr(sys, "stdin", NoFileno())
    assert _stdin_fd() == -1
    assert TerminalKeys()._fd == -1

    class CapturedStdin:
        def fileno(self):
            raise ValueError("I/O operation on captured stream")

    monkeypatch.setattr(sys, "stdin", CapturedStdin())
    assert _stdin_fd() == -1

    class RealStdin:
        def fileno(self):
            return 42

    monkeypatch.setattr(sys, "stdin", RealStdin())
    assert _stdin_fd() == 42
