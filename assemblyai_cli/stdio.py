from __future__ import annotations

import sys
from collections.abc import Iterator


def stdin_is_piped() -> bool:
    """True when stdin is a pipe/redirect rather than an interactive terminal."""
    stream = sys.stdin
    return stream is not None and not stream.isatty()


def iter_piped_stdin_lines() -> Iterator[str]:
    """Yield non-blank, stripped lines piped on stdin, live, as each one arrives.

    Unlike ``piped_stdin_text`` (which reads to EOF), this consumes the pipe
    incrementally so a long-running upstream like ``aai stream -o text`` can drive
    a downstream command turn by turn. Yields nothing when stdin is a terminal, so
    a ``--follow`` consumer used interactively returns instead of blocking forever.
    """
    stream = sys.stdin
    if stream is None or stream.isatty():
        return
    for raw in stream:
        line = raw.strip()
        if line:
            yield line


def piped_stdin_text() -> str | None:
    """Return text piped on stdin, or None when stdin is a terminal or empty.

    Lets commands accept input from a pipe (e.g. ``cat notes.txt | aai llm ...`` or
    ``aai transcribe x.mp3 -o text | aai llm "summarize"``) without blocking when run
    interactively.
    """
    stream = sys.stdin
    if stream is None or stream.isatty():
        return None
    data = stream.read()
    return data if data.strip() else None


def read_binary_stdin() -> bytes:
    """Read all bytes piped on stdin, for a ``-`` audio source.

    Used by ``cat call.wav | aai transcribe -`` and ``ffmpeg … | aai transcribe -``.
    """
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # e.g. a text-only stub in tests
        return sys.stdin.read().encode() if sys.stdin is not None else b""
    return bytes(buffer.read())
