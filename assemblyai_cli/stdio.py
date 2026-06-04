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
