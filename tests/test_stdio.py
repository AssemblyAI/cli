import io

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
