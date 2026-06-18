from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Generator, Iterator

# The OS-level stderr descriptor. Used as a literal rather than ``sys.stderr.fileno()``
# because inside a Textual app ``sys.stderr`` is swapped for a redirector whose ``fileno()``
# returns an unusable fd — duping that raised EBADF and broke the very mic open we were
# trying to quiet. fd 2 is the real inherited stderr, which Textual never touches.
_STDERR_FD = 2


@contextlib.contextmanager
def suppress_native_stderr() -> Generator[None]:
    """Send OS-level stderr to /dev/null for the block, then restore it.

    Catches diagnostics that C extensions write straight to fd 2 via ``fprintf`` —
    PortAudio/CoreAudio/ALSA print device-probe noise there on the first audio call,
    *below* Python's logging, so silencing loggers can't reach it. Inside a full-screen
    TUI (which draws to stdout and never repaints stderr) those raw writes scribble over
    the rendered screen; the mic-open path wraps its PortAudio calls in this so they land
    in the void instead.

    Safe by construction: if the descriptor can't be duplicated/redirected for any reason,
    the block runs with stderr untouched rather than raising — suppression is cosmetic and
    must never break the operation it wraps. Exceptions from the body propagate normally
    (only the fd is redirected, not raised errors).
    """
    saved_fd: int | None = None
    devnull_fd: int | None = None
    try:
        saved_fd = os.dup(_STDERR_FD)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, _STDERR_FD)
    except OSError:
        # Couldn't redirect — abandon suppression, never break the caller.
        _close_quietly(saved_fd)
        _close_quietly(devnull_fd)
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.dup2(saved_fd, _STDERR_FD)  # restore the real stderr
        _close_quietly(saved_fd)
        _close_quietly(devnull_fd)


def _close_quietly(fd: int | None) -> None:
    """Close ``fd`` if it was opened, ignoring an already-closed/invalid descriptor."""
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)


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


def stdin_is_tty() -> bool:
    """True when stdin is an interactive terminal. The single raw-``isatty`` chokepoint
    for stdin, so higher layers compose this rather than re-reaching for ``sys.stdin``."""
    return sys.stdin.isatty()


def stdout_is_tty() -> bool:
    """True when stdout is an interactive terminal. The single raw-``isatty`` chokepoint
    for stdout (e.g. `output.is_agentic`/`print_code` compose it)."""
    return sys.stdout.isatty()


def stderr_is_tty() -> bool:
    """True when stderr is an interactive terminal. The single raw-``isatty`` chokepoint
    for stderr (the browser-login interactivity probe composes it)."""
    return sys.stderr.isatty()


def interactive_stdio() -> bool:
    """True only when stdin and stdout are both real TTYs — i.e. a human can answer
    a prompt and see it. The shared "may we prompt here?" predicate for the bare-`assembly`
    setup offer, the onboarding prompter, and the `assembly init` template picker, so the
    three can't drift on what counts as interactive.
    """
    return stdin_is_tty() and stdout_is_tty()


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
