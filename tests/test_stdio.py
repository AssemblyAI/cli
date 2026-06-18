import io
import os

from aai_cli.core import stdio


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _Pipe(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_iter_piped_stdin_lines_yields_stripped_nonblank(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe("alpha\n\n  \nbeta\n"))
    assert list(stdio.iter_piped_stdin_lines()) == ["alpha", "beta"]


def test_iter_piped_stdin_lines_empty_on_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty("alpha\nbeta\n"))
    assert list(stdio.iter_piped_stdin_lines()) == []


def test_stdin_is_piped(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe(""))
    assert stdio.stdin_is_piped() is True
    monkeypatch.setattr("sys.stdin", _Tty(""))
    assert stdio.stdin_is_piped() is False


def test_stdin_is_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty(""))
    assert stdio.stdin_is_tty() is True
    monkeypatch.setattr("sys.stdin", _Pipe(""))
    assert stdio.stdin_is_tty() is False


def test_stdout_is_tty(monkeypatch):
    monkeypatch.setattr("sys.stdout", _Tty(""))
    assert stdio.stdout_is_tty() is True
    monkeypatch.setattr("sys.stdout", _Pipe(""))
    assert stdio.stdout_is_tty() is False


def test_stderr_is_tty(monkeypatch):
    monkeypatch.setattr("sys.stderr", _Tty(""))
    assert stdio.stderr_is_tty() is True
    monkeypatch.setattr("sys.stderr", _Pipe(""))
    assert stdio.stderr_is_tty() is False


def test_interactive_stdio_requires_both_stdin_and_stdout_tty(monkeypatch):
    # Both must be terminals; either one piped flips it false (kills the and->or mutant).
    monkeypatch.setattr("sys.stdin", _Tty(""))
    monkeypatch.setattr("sys.stdout", _Tty(""))
    assert stdio.interactive_stdio() is True
    monkeypatch.setattr("sys.stdout", _Pipe(""))
    assert stdio.interactive_stdio() is False
    monkeypatch.setattr("sys.stdin", _Pipe(""))
    monkeypatch.setattr("sys.stdout", _Tty(""))
    assert stdio.interactive_stdio() is False


def test_piped_stdin_text_returns_none_on_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Tty("ignored\n"))
    assert stdio.piped_stdin_text() is None


def test_piped_stdin_text_returns_none_when_blank(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe("   \n"))
    assert stdio.piped_stdin_text() is None


def test_piped_stdin_text_returns_text(monkeypatch):
    monkeypatch.setattr("sys.stdin", _Pipe("hello world\n"))
    assert stdio.piped_stdin_text() == "hello world\n"


def test_read_binary_stdin_uses_buffer(monkeypatch):
    # A real stdin exposes a binary `.buffer`; read_binary_stdin reads from it.
    class _WithBuffer:
        def __init__(self, data: bytes):
            self.buffer = io.BytesIO(data)

    monkeypatch.setattr("sys.stdin", _WithBuffer(b"\x00\x01\x02"))
    assert stdio.read_binary_stdin() == b"\x00\x01\x02"


def test_read_binary_stdin_falls_back_for_text_only_stub(monkeypatch):
    # A text-only stub (no .buffer) — e.g. CliRunner's StringIO in tests.
    monkeypatch.setattr("sys.stdin", _Pipe("abc"))
    assert stdio.read_binary_stdin() == b"abc"


def test_silence_stdout_redirects_to_devnull(monkeypatch):
    calls = {}

    def fake_open(path, flags):
        calls["path"] = path
        return 99

    def fake_dup2(fd_src, fd_dst):
        calls["dup2"] = (fd_src, fd_dst)

    monkeypatch.setattr("os.open", fake_open)
    monkeypatch.setattr("os.dup2", fake_dup2)
    monkeypatch.setattr("os.close", lambda fd: calls.setdefault("closed", fd))
    stdio.silence_stdout()
    assert calls["path"] == __import__("os").devnull
    assert calls["dup2"][0] == 99
    # The temporary devnull fd must be closed after dup2 — it leaks otherwise.
    assert calls["closed"] == 99


def test_silence_stdout_suppresses_oserror(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("no fd")

    # Raising inside the suppressed block must not propagate.
    monkeypatch.setattr("os.open", boom)
    stdio.silence_stdout()


def test_suppress_native_stderr_redirects_during_block_then_restores(monkeypatch):
    # The fd dance: dup the real stderr (fd 2 itself — never sys.stderr.fileno(), which is
    # an unusable redirector inside a TUI), point it at /dev/null for the body, then restore
    # and close both temporaries. The body must run *while* redirected (between the dup2s).
    events: list[object] = []
    monkeypatch.setattr("os.dup", lambda fd: events.append(("dup", fd)) or 50)
    monkeypatch.setattr("os.open", lambda path, flags: events.append(("open", path)) or 99)
    monkeypatch.setattr("os.dup2", lambda src, dst: events.append(("dup2", src, dst)))
    monkeypatch.setattr("os.close", lambda fd: events.append(("close", fd)))

    with stdio.suppress_native_stderr():
        events.append("body")

    assert events == [
        ("dup", 2),  # save the real stderr fd (literal 2)
        ("open", os.devnull),  # open /dev/null
        ("dup2", 99, 2),  # point stderr at /dev/null
        "body",  # the block runs while stderr is redirected
        ("dup2", 50, 2),  # restore the saved fd
        ("close", 50),
        ("close", 99),
    ]


def test_suppress_native_stderr_runs_body_when_redirect_fails(monkeypatch):
    # Safe by construction: if the fd can't be duplicated, the block still runs (suppression
    # is cosmetic and must never break the wrapped mic open) and stderr is never redirected.
    def boom(_fd: int) -> int:
        raise OSError("cannot dup")

    redirected: list[tuple[int, int]] = []
    monkeypatch.setattr("os.dup", boom)
    monkeypatch.setattr("os.dup2", lambda src, dst: redirected.append((src, dst)))
    ran: list[bool] = []

    with stdio.suppress_native_stderr():
        ran.append(True)

    assert ran == [True]  # body ran despite the dup failure
    assert redirected == []  # never redirected -> nothing left to restore


def test_suppress_native_stderr_swallows_close_failure(monkeypatch):
    # A teardown close hitting an already-closed/invalid fd must not escape the block.
    def boom(_fd: int) -> None:
        raise OSError("already closed")

    monkeypatch.setattr("os.dup", lambda _fd: 50)
    monkeypatch.setattr("os.open", lambda _path, _flags: 99)
    monkeypatch.setattr("os.dup2", lambda _src, _dst: None)
    monkeypatch.setattr("os.close", boom)

    with stdio.suppress_native_stderr():
        pass  # exits cleanly even though both teardown closes raise
