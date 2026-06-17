"""Tests for the shared media-batch runner (aai_cli/app/batch.py).

Drives :func:`batch.run_batch` with fake workers (no command pipeline) so the
concurrency, NDJSON, skip, failure, and summary behavior is exercised directly,
and :func:`batch.stdin_sources` with a stubbed stdin reader.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from aai_cli.app import batch
from aai_cli.core import stdio
from aai_cli.core.errors import CLIError, NotAuthenticated, UsageError


def _completed(source: str) -> batch.SourceResult:
    return batch.SourceResult(
        payload={"source": source, "out": f"{source}.out"}, summary=f"{source} done"
    )


def test_source_result_is_immutable():
    result = batch.SourceResult(payload={}, summary="x")
    # A computed field name (not a string literal) so ruff B010 doesn't rewrite the
    # setattr back into a static assignment pyright would then reject.
    field_name = dataclasses.fields(result)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result, field_name, None)


# --- stdin_sources -------------------------------------------------------------


def test_stdin_sources_returns_none_without_the_flag():
    assert batch.stdin_sources("talk.mp4", from_stdin=False) is None


def test_stdin_sources_reads_and_dedupes_stdin_lines(monkeypatch):
    monkeypatch.setattr(stdio, "iter_piped_stdin_lines", lambda: iter(["a.mp4", "b.mp4", "a.mp4"]))
    assert batch.stdin_sources("", from_stdin=True) == ["a.mp4", "b.mp4"]


def test_stdin_sources_rejects_a_positional_source_with_the_flag():
    with pytest.raises(UsageError) as exc:
        batch.stdin_sources("talk.mp4", from_stdin=True)
    assert "don't also pass a source" in exc.value.message


def test_stdin_sources_rejects_empty_stdin(monkeypatch):
    monkeypatch.setattr(stdio, "iter_piped_stdin_lines", lambda: iter([]))
    with pytest.raises(UsageError) as exc:
        batch.stdin_sources("", from_stdin=True)
    assert "No sources received on stdin" in exc.value.message


# --- run_batch: human mode -----------------------------------------------------


def test_run_batch_human_renders_a_table_and_a_summary(capsys):
    batch.run_batch(
        ["a.mp4", "b.mp4"],
        worker=_completed,
        concurrency=2,
        summary_verb="Captioned",
        json_mode=False,
        quiet=False,
    )
    captured = capsys.readouterr()
    # The table (stdout) lists every source and its result line.
    assert "a.mp4" in captured.out
    assert "b.mp4" in captured.out
    assert "done" in captured.out
    # The tally lands on stderr so stdout stays the result surface.
    assert "Captioned 2, skipped 0." in captured.err
    # Human mode emits no JSON.
    assert "{" not in captured.out


def test_run_batch_summary_counts_skips(capsys):
    def worker(source: str) -> batch.SourceResult:
        if source == "a.mp4":
            return batch.SourceResult(
                payload={"source": source}, summary="exists", status="skipped"
            )
        return _completed(source)

    batch.run_batch(
        ["a.mp4", "b.mp4"],
        worker=worker,
        concurrency=1,
        summary_verb="Dubbed",
        json_mode=False,
        quiet=False,
    )
    assert "Dubbed 1, skipped 1." in capsys.readouterr().err


def test_run_batch_quiet_suppresses_the_summary(capsys):
    batch.run_batch(
        ["a.mp4"],
        worker=_completed,
        concurrency=1,
        summary_verb="Clipped",
        json_mode=False,
        quiet=True,
    )
    assert "Clipped" not in capsys.readouterr().err


# --- run_batch: JSON mode ------------------------------------------------------


def test_run_batch_json_emits_one_ndjson_record_per_source(capsys):
    batch.run_batch(
        ["a.mp4", "b.mp4"],
        worker=_completed,
        concurrency=1,
        summary_verb="Captioned",
        json_mode=True,
        quiet=False,
    )
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert len(records) == 2
    assert {r["source"] for r in records} == {"a.mp4", "b.mp4"}
    assert all(r["type"] == "result" and r["status"] == "completed" for r in records)
    # The completed payload is merged in (the command's own JSON shape).
    assert {r["out"] for r in records} == {"a.mp4.out", "b.mp4.out"}
    # JSON mode prints no human summary on stderr.
    assert "Captioned" not in captured.err


def test_run_batch_json_records_a_skip_with_its_payload(capsys):
    def worker(source: str) -> batch.SourceResult:
        return batch.SourceResult(
            payload={"source": source, "out": "x"}, summary="exists", status="skipped"
        )

    batch.run_batch(
        ["a.mp4"], worker=worker, concurrency=1, summary_verb="Dubbed", json_mode=True, quiet=False
    )
    record = json.loads(capsys.readouterr().out)
    assert record == {"type": "result", "source": "a.mp4", "status": "skipped", "out": "x"}


def test_run_batch_json_records_a_failure_as_an_error(capsys):
    def worker(source: str) -> batch.SourceResult:
        if source == "bad.mp4":
            raise CLIError("boom", error_type="x")
        return _completed(source)

    with pytest.raises(CLIError) as exc:
        batch.run_batch(
            ["bad.mp4", "ok.mp4"],
            worker=worker,
            concurrency=1,
            summary_verb="Captioned",
            json_mode=True,
            quiet=False,
        )
    # The batch still exits non-zero, with a resume hint.
    assert "1 of 2 sources failed" in exc.value.message
    assert exc.value.error_type == "batch_failed"
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failed = next(r for r in records if r["status"] == "failed")
    assert failed == {"type": "result", "source": "bad.mp4", "status": "failed", "error": "boom"}


# --- run_batch: failure handling -----------------------------------------------


def test_run_batch_raises_when_any_source_fails(capsys):
    def worker(source: str) -> batch.SourceResult:
        raise CLIError("nope", error_type="x")

    with pytest.raises(CLIError) as exc:
        batch.run_batch(
            ["a.mp4"], worker=worker, concurrency=1, summary_verb="X", json_mode=False, quiet=False
        )
    assert exc.value.error_type == "batch_failed"
    # A failed batch never prints the success tally.
    assert "skipped" not in capsys.readouterr().err


def test_run_batch_aborts_the_whole_batch_on_not_authenticated():
    def worker(source: str) -> batch.SourceResult:
        raise NotAuthenticated("sign in")

    with pytest.raises(NotAuthenticated):
        batch.run_batch(
            ["a.mp4", "b.mp4"],
            worker=worker,
            concurrency=1,
            summary_verb="X",
            json_mode=False,
            quiet=False,
        )
