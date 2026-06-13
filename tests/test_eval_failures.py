"""`assembly eval` concurrency and failure-resilience behavior.

Split out of test_eval_command.py to keep modules under the 500-line gate.
A bad row must not discard the other (paid) rows; only a rejected key or a
non-CLIError bug aborts the run (cancelling queued items in concurrent mode).
"""

import contextlib
import json
import threading
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.commands.evaluate import _exec as evaluate
from aai_cli.core.errors import APIError, auth_failure
from aai_cli.main import app
from tests.test_eval_command import (
    _auth,
    _mock_transcribe,
    _payload_of,
    _transcript,
    _write_wer_manifest,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_concurrency_runs_items_at_once_and_keeps_dataset_order(tmp_path, mocker):
    # With concurrency 2 and two items, both transcriptions must be in flight at
    # once (the barrier times out otherwise), and the rows stay in dataset order
    # no matter which finishes first.
    _auth()
    _write_wer_manifest(tmp_path)
    barrier = threading.Barrier(2, timeout=10)
    texts = {"a.wav": "hello there", "b.wav": "goodbye cow"}

    def fake_transcribe(api_key, audio, *, config):
        barrier.wait()
        return _transcript(texts[Path(audio).name])

    mocker.patch(
        "aai_cli.commands.evaluate._exec.client.transcribe",
        autospec=True,
        side_effect=fake_transcribe,
    )
    result = runner.invoke(app, ["eval", "manifest.csv", "--concurrency", "2", "--json"])
    assert result.exit_code == 0
    payload = _payload_of(result)
    assert [row["item"] for row in payload["rows"]] == ["a.wav", "b.wav"]
    assert payload["rows"][0]["wer"] == 0.0
    assert payload["rows"][1]["wer"] == 0.5


def test_concurrency_shows_one_pooled_status(tmp_path, mocker, monkeypatch):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), _transcript("goodbye now")])
    seen = []

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet):
        seen.append(message)
        yield

    monkeypatch.setattr("aai_cli.commands.evaluate._exec.output.status", fake_status)
    assert runner.invoke(app, ["eval", "manifest.csv", "--concurrency", "2"]).exit_code == 0
    assert seen == ["Transcribing 2 items (concurrency 2)…"]


class _CapturingPool:
    """Wrap ThreadPoolExecutor so a test can see how shutdown was called."""

    def __init__(self, monkeypatch):
        self.seen = {}
        real = evaluate.ThreadPoolExecutor
        capture = self

        class Capture(real):
            def shutdown(self, wait=True, *, cancel_futures=False):
                capture.seen.setdefault("cancel_futures", cancel_futures)  # first call wins
                super().shutdown(wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(evaluate, "ThreadPoolExecutor", Capture)


def test_concurrent_row_failure_keeps_other_rows_and_exits_nonzero(tmp_path, mocker, monkeypatch):
    # A per-row APIError must not cancel the pool: the other (paid) rows finish
    # and the run exits 1 with the failure tally, same as sequential mode.
    _auth()
    _write_wer_manifest(tmp_path)
    pool = _CapturingPool(monkeypatch)
    _mock_transcribe(mocker, [_transcript("hello there"), APIError("rate limited")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--concurrency", "2", "--json"])
    assert result.exit_code == 1
    assert "1 of 2 items failed" in result.output
    assert pool.seen["cancel_futures"] is False
    payload = _payload_of(result)
    assert payload["failed"] == 1
    errored = [row for row in payload["rows"] if "error" in row]
    assert len(errored) == 1 and "rate limited" in errored[0]["error"]
    assert any("wer" in row for row in payload["rows"])  # the good row was still scored


def test_concurrent_rejected_key_drops_queued_items(tmp_path, mocker, monkeypatch):
    # One rejected key fails every row identically, so the abort path must shut
    # the pool down with cancel_futures=True — that's what keeps it from burning
    # an API call per queued item. Asserted on the shutdown call because which
    # queued futures actually get dropped is a race (the test_transcribe_batch.py
    # pattern).
    _auth()
    _write_wer_manifest(tmp_path)
    pool = _CapturingPool(monkeypatch)
    _mock_transcribe(mocker, [auth_failure(), auth_failure()])
    result = runner.invoke(app, ["eval", "manifest.csv", "--concurrency", "2"])
    assert result.exit_code == 4
    assert pool.seen["cancel_futures"] is True


def test_concurrent_internal_bug_drops_queued_items(tmp_path, mocker, monkeypatch):
    # A non-CLIError is a bug, not a row outcome: re-raised (internal error),
    # with queued items cancelled.
    _auth()
    _write_wer_manifest(tmp_path)
    pool = _CapturingPool(monkeypatch)
    _mock_transcribe(mocker, [RuntimeError("boom"), RuntimeError("boom")])
    result = runner.invoke(app, ["eval", "manifest.csv", "--concurrency", "2"])
    assert result.exit_code == 1
    assert "boom" in result.output
    assert pool.seen["cancel_futures"] is True


def test_concurrency_below_one_is_a_usage_error(tmp_path):
    result = runner.invoke(app, ["eval", "org/ds", "--concurrency", "0"])
    assert result.exit_code == 2


def test_api_error_mid_run_fails_cleanly(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), APIError("rate limited")])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert result.exit_code == 1
    assert "rate limited" in result.output


def _write_three_row_manifest(tmp_path):
    for name in ("a.wav", "b.wav", "c.wav"):
        (tmp_path / name).write_bytes(b"fake-audio")
    (tmp_path / "manifest.csv").write_text(
        "audio,text\na.wav,hello there\nb.wav,goodbye now\nc.wav,see you\n", encoding="utf-8"
    )


def test_failed_row_keeps_completed_rows_and_summary_pools_scored_only(tmp_path, mocker):
    # One bad row must not discard the completed (paid) rows: the run keeps going,
    # the summary pools only the scored rows, and the exit code is nonzero.
    _auth()
    _write_three_row_manifest(tmp_path)
    _mock_transcribe(
        mocker,
        [_transcript("hello there"), APIError("rate limited"), _transcript("see you")],
    )
    result = runner.invoke(app, ["eval", "manifest.csv", "--json"])
    assert result.exit_code == 1
    payload = _payload_of(result)
    assert payload["items"] == 3
    assert payload["failed"] == 1
    assert payload["rows"][0] == {"item": "a.wav", "words": 2, "errors": 0, "wer": 0.0}
    assert payload["rows"][1] == {"item": "b.wav", "error": "rate limited"}
    assert payload["rows"][2] == {"item": "c.wav", "words": 2, "errors": 0, "wer": 0.0}
    # Pooled over the two scored rows only — the failed row contributes no words.
    assert payload["words"] == 4
    assert payload["errors"] == 0
    assert payload["wer"] == 0.0
    err = next(
        json.loads(line) for line in result.output.splitlines() if line.startswith('{"error"')
    )
    assert err["error"]["type"] == "eval_failed"
    assert "1 of 3 items failed" in err["error"]["message"]


def test_failed_row_renders_error_column_in_human_table(tmp_path, mocker):
    _auth()
    _write_wer_manifest(tmp_path)
    _mock_transcribe(mocker, [_transcript("hello there"), APIError("rate limited")])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert result.exit_code == 1
    assert "ERROR" in result.output  # the failure column appears
    assert "rate limited" in result.output  # with the row's error
    assert "0.00%" in result.output  # and the scored row still renders
    assert "1 of 2 items failed" in result.output


def test_rejected_key_aborts_eval_with_auth_exit_code(tmp_path, mocker):
    # A rejected key fails every row identically — abort instead of billing through
    # the whole dataset, and keep the auth exit code (4), not the eval-failed 1.
    _auth()
    _write_wer_manifest(tmp_path)
    tx = _mock_transcribe(mocker, [auth_failure()])
    result = runner.invoke(app, ["eval", "manifest.csv"])
    assert result.exit_code == 4
    assert tx.call_count == 1  # aborted on the first row, no further billing
    assert "rejected" in result.output


def test_unauthenticated_fails_before_dataset_download(mocker):
    # Credentials resolve before the dataset loads: a signed-out user must not
    # pull the whole dataset first.
    load = mocker.patch("aai_cli.commands.evaluate._exec.eval_data.load", autospec=True)
    result = runner.invoke(app, ["eval", "org/ds"])
    assert result.exit_code == 4
    load.assert_not_called()
