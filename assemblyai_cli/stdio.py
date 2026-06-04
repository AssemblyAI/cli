from __future__ import annotations

import sys


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
