"""Concurrent many-source batch mode for the media commands (clip, dub, caption).

``assembly transcribe`` owns a richer batch path — per-source ``.aai.json``
sidecars, resumable ``--llm`` chains, ``--llm-reduce`` (see
``app/transcribe/batch.py``). This is the lighter shared machinery for the
one-source-in / one-output-out media commands: a command reads its source list
with :func:`stdin_sources` and, when batch mode is active, hands each source to
:func:`run_batch` with a worker that runs its existing single-source path.

The worker returns a :class:`SourceResult` (or raises a ``CLIError`` for a
per-source failure). The runner renders one live status table, emits one NDJSON
record per source under ``--json``, and raises so the process exits non-zero if
any source failed — a re-run resumes (each command skips a source whose output
already exists unless ``--force``) and retries only the failures.

Source expansion stays deliberately small: stdin only (one path/URL per line).
Directory/glob/feed expansion is transcribe's concern, not these commands'.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Generator
from contextlib import contextmanager

from rich.live import Live
from rich.markup import escape
from rich.table import Table

from aai_cli.core import stdio
from aai_cli.core.errors import CLIError, NotAuthenticated, UsageError
from aai_cli.ui import output, theme

# A worker turns one source into its result, or raises a CLIError to fail it.
Worker = Callable[[str], "SourceResult"]


@dataclasses.dataclass(frozen=True)
class SourceResult:
    """One source's outcome: the JSON payload the command would emit single-source,
    plus a one-line human ``summary`` for the progress table. ``status`` is
    ``"completed"`` for processed sources or ``"skipped"`` for ones a re-run left
    alone (their output already existed and ``--force`` wasn't passed)."""

    payload: dict[str, object]
    summary: str
    status: str = "completed"


def stdin_sources(media: str, *, from_stdin: bool) -> list[str] | None:
    """The batch source list read from stdin, or ``None`` for a single-source run.

    ``None`` means "not batch mode" (no ``--from-stdin``); the caller then handles
    its lone ``media`` argument as before. With ``--from-stdin`` a positional source
    is rejected (the list comes from stdin, not argv), and an empty stdin is a usage
    error rather than a silent no-op.
    """
    if not from_stdin:
        return None
    if media:
        raise UsageError(
            "--from-stdin reads sources from stdin; don't also pass a source.",
            suggestion="Drop the positional source, or drop --from-stdin to process just it.",
        )
    lines = list(dict.fromkeys(stdio.iter_piped_stdin_lines()))  # dedupe, keep order
    if not lines:
        raise UsageError(
            "No sources received on stdin.",
            suggestion="Pipe one path or URL per line, e.g. "
            "find . -name '*.mp4' | assembly caption --from-stdin.",
        )
    return lines


@dataclasses.dataclass
class _Item:
    source: str
    status: str = "queued"  # queued → processing → completed | skipped | failed
    summary: str = ""  # the result one-liner, or the error message when failed
    payload: dict[str, object] | None = None

    def record(self) -> dict[str, object]:
        """The NDJSON record emitted for this source under ``--json``."""
        # "type" discriminates NDJSON lines CLI-wide (see REFERENCE.md "JSON output").
        rec: dict[str, object] = {"type": "result", "source": self.source, "status": self.status}
        if self.status == "failed":
            rec["error"] = self.summary
        elif self.payload is not None:
            rec.update(self.payload)
        return rec


def _process_one(item: _Item, worker: Worker) -> None:
    """Worker body: run one source, recording its result or per-source failure.

    A NotAuthenticated re-raises so :func:`_drain` aborts the whole batch — one
    rejected key fails every source identically — while any other CLIError is
    recorded on the item and the batch carries on.
    """
    try:
        item.status = "processing"
        result = worker(item.source)
        item.status, item.summary, item.payload = result.status, result.summary, result.payload
    except CLIError as err:
        item.status, item.summary = "failed", err.message
        if isinstance(err, NotAuthenticated):
            raise


def _render_table(items: list[_Item]) -> Table:
    table = output.data_table("Source", "Status", "Result")
    for item in items:
        table.add_row(escape(item.source), theme.status_text(item.status), escape(item.summary))
    return table


@contextmanager
def _progress_table(items: list[_Item], *, json_mode: bool) -> Generator[None]:
    """Render the batch as a live-updating table (human mode); a no-op under ``--json``.

    Rich renders nothing while running on a non-interactive console and prints the
    final frame once on stop, so piped/agent runs still get the result table. JSON
    mode skips Rich entirely — NDJSON per source is the output.
    """
    if json_mode:
        yield
        return
    with Live(
        get_renderable=lambda: _render_table(items),
        console=output.console,
        refresh_per_second=4,  # pragma: no mutate (cosmetic refresh cadence)
    ):
        yield


def _drain(items: list[_Item], *, worker: Worker, concurrency: int, json_mode: bool) -> None:
    """Run the workers, emitting one NDJSON record per finished source under ``--json``.

    The first exception that escapes a worker (NotAuthenticated, or a bug) drops the
    not-yet-started sources and re-raises once the in-flight ones drain.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_one, item, worker): item for item in items}
        for future in as_completed(futures):
            if (exc := future.exception()) is not None:
                pool.shutdown(cancel_futures=True)  # pragma: no mutate (best-effort cleanup)
                raise exc
            if json_mode:
                output.emit_ndjson(futures[future].record())


def _summarize(items: list[_Item], *, summary_verb: str, json_mode: bool, quiet: bool) -> None:
    """Report the batch tally, raising so a partly-failed batch exits non-zero."""
    failed = sum(1 for item in items if item.status == "failed")
    if failed:
        raise CLIError(
            f"{failed} of {len(items)} sources failed.",
            error_type="batch_failed",
            suggestion="Re-run the same command to retry only the failures — "
            "sources whose output already exists are skipped.",
        )
    completed = sum(1 for item in items if item.status == "completed")
    skipped = len(items) - completed
    if not json_mode and not quiet:
        output.error_console.print(
            output.success(f"{summary_verb} {completed}, skipped {skipped}.")
        )


def run_batch(
    sources: list[str],
    *,
    worker: Worker,
    concurrency: int,
    summary_verb: str,
    json_mode: bool,
    quiet: bool,
) -> None:
    """Process ``sources`` concurrently through ``worker``, one output per source.

    Raises CLIError (exit 1) when any source failed so scripts can trust the exit
    code; a re-run resumes (finished outputs are skipped) and retries the failures.
    """
    items = [_Item(source) for source in sources]
    with _progress_table(items, json_mode=json_mode):
        _drain(items, worker=worker, concurrency=concurrency, json_mode=json_mode)
    _summarize(items, summary_verb=summary_verb, json_mode=json_mode, quiet=quiet)
