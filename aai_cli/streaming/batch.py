"""Drive a ``assembly stream --from-stdin`` list of sources, one realtime session each.

The realtime API is one session at a time, so a list of files/URLs (read on stdin,
one per line) streams sequentially. This lives beside ``StreamSession`` rather than
inside it: a session owns *one* run, while this owns the sequence — fresh session per
source, per-source failure accounting, and the batch-wide Ctrl-C/pipe handling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import typer

from aai_cli.core.errors import CLIError, NotAuthenticated
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.session import StreamSession
from aai_cli.ui import output

# A batch source string resolved to its real-time audio chunks and declared rate.
_OpenedSource = tuple[Iterable[bytes], int]


def _stream_source(
    source: str,
    *,
    index: int,
    total: int,
    make_session: Callable[[], StreamSession],
    open_source: Callable[[str], _OpenedSource],
    renderer: StreamRenderer,
    json_mode: bool,
) -> bool:
    """Stream one batch source in its own session; return True when it failed.

    A ``CLIError`` (bad path, missing ffmpeg, decode failure) is recorded as a warning
    so the batch carries on — except ``NotAuthenticated``, which re-raises to abort the
    whole batch (one rejected key fails every source identically, and auto-login should
    trigger once).
    """
    renderer.source(source, index=index, total=total)
    try:
        audio, rate = open_source(source)
        make_session().run(audio, rate, handle_interrupt=False)
    except NotAuthenticated:
        raise
    except CLIError as exc:
        # Flatten newlines so a crafted path/URL can't inject extra log lines (CR/LF).
        detail = f"{source}: {exc.message}".replace("\n", " ").replace("\r", " ")
        output.emit_warning(detail, json_mode=json_mode)
        return True
    else:
        return False


def stream_batch_sources(
    sources: list[str],
    *,
    make_session: Callable[[], StreamSession],
    open_source: Callable[[str], _OpenedSource],
    renderer: StreamRenderer,
    json_mode: bool,
) -> None:
    """Stream each source in ``sources`` in turn — the ``assembly stream --from-stdin``
    batch mode.

    The realtime API is one session at a time, so a list of files/URLs streams
    sequentially: each source gets a fresh ``StreamSession`` from ``make_session`` (its
    own transcript and ``--llm`` chain state) via ``_stream_source``.

    A Ctrl-C or a closed downstream pipe stops the batch cleanly (exit 0). When any
    source failed, raises a ``CLIError`` at the end so a script can trust the exit code.
    """
    total = len(sources)
    failures = 0
    try:
        for index, source in enumerate(sources, start=1):
            failures += _stream_source(
                source,
                index=index,
                total=total,
                make_session=make_session,
                open_source=open_source,
                renderer=renderer,
                json_mode=json_mode,
            )
    except KeyboardInterrupt:
        # One Ctrl-C stops the whole batch, not just the current source -> exit 0.
        renderer.stopped()
        return
    except BrokenPipeError:
        # Downstream consumer (e.g. `| head`) closed the pipe; stop quietly.
        raise typer.Exit(code=0) from None
    finally:
        renderer.close()
    if failures:
        raise CLIError(
            f"{failures} of {total} sources failed.",
            error_type="batch_failed",
            suggestion="Check each failed path or URL, then re-run.",
        )
