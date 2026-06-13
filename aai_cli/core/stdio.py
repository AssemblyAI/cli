from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator


def silence_stdout() -> None:
    """Point stdout at /dev/null so a closed downstream pipe can't re-raise.

    Once a consumer (e.g. ``| head``) closes the pipe, a later write — or Python's
    flush at interpreter shutdown — raises BrokenPipeError with an ugly "Exception
    ignored" traceback. Redirecting the fd makes those writes no-ops. Shared by the
    one-shot entry point and the streaming reader thread.
    """
    with contextlib.suppress(OSError):
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, sys.stdout.fileno())
        finally:
            # dup2 duplicates the descriptor, so the original must be closed
            # or it leaks one fd per call.
            os.close(devnull_fd)


def interactive_stdio() -> bool:
    """True only when stdin and stdout are both real TTYs — i.e. a human can answer
    a prompt and see it. The shared "may we prompt here?" predicate for the bare-`assembly`
    setup offer, the onboarding prompter, and the `assembly init` template picker, so the
    three can't drift on what counts as interactive.
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def stdin_is_piped() -> bool:
    """True when stdin is a pipe/redirect rather than an interactive terminal."""
    stream = sys.stdin
    return stream is not None and not stream.isatty()


def iter_piped_stdin_lines() -> Iterator[str]:
    """Yield non-blank, stripped lines piped on stdin, live, as each one arrives.

    Unlike ``piped_stdin_text`` (which reads to EOF), this consumes the pipe
    incrementally so a long-running upstream like ``assembly stream -o text`` can drive
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

    Lets commands accept input from a pipe (e.g. ``cat notes.txt | assembly llm ...`` or
    ``assembly transcribe x.mp3 -o text | assembly llm "summarize"``) without blocking when run
    interactively.
    """
    stream = sys.stdin
    if stream is None or stream.isatty():
        return None
    data = stream.read()
    return data if data.strip() else None


def read_binary_stdin() -> bytes:
    """Read all bytes piped on stdin, for a ``-`` audio source.

    Used by ``cat call.wav | assembly transcribe -`` and ``ffmpeg … | assembly transcribe -``.
    """
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is None:  # e.g. a text-only stub in tests
        return sys.stdin.read().encode() if sys.stdin is not None else b""
    return bytes(buffer.read())
