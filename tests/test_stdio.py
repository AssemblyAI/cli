import io

from assemblyai_cli import stdio


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
