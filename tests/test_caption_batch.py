"""Batch-mode tests for `assembly caption` (--from-stdin / --concurrency / --force).

The single-source pipeline (`_caption_one` → transcribe → SRT → ffmpeg) is covered
in test_caption_exec.py; here `_caption_one` is faked so the tests pin only the
batch wiring: source dispatch, the skip-on-existing-output worker, and the
single-result flags that can't span a batch.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from aai_cli.app import batch
from aai_cli.app.context import AppState
from aai_cli.commands.caption import _exec as caption_exec
from aai_cli.core import stdio
from aai_cli.core.errors import UsageError
from tests.test_caption_exec import DEFAULTS


def _pipe(monkeypatch: pytest.MonkeyPatch, *sources: str) -> None:
    monkeypatch.setattr(stdio, "iter_piped_stdin_lines", lambda: iter(sources))


def _fake_caption_one(seen: list[str]):
    def one(opts, state, *, json_mode):
        # The batch worker silences spinners by handing each source a quiet state
        # (concurrent Rich spinners would clobber the live table).
        assert state.quiet is True
        seen.append(opts.media)
        return batch.SourceResult(
            payload={"source": opts.media, "out": f"{opts.media}.captioned"},
            summary=f"{opts.media} captioned",
        )

    return one


def _batch_opts(**overrides):
    return dataclasses.replace(DEFAULTS, media="", from_stdin=True, **overrides)


# --- source dispatch -----------------------------------------------------------


def test_batch_captions_each_piped_source(monkeypatch, capsys):
    _pipe(monkeypatch, "a.mp4", "b.mp4")
    seen: list[str] = []
    monkeypatch.setattr(caption_exec, "_caption_one", _fake_caption_one(seen))

    caption_exec.run_caption(_batch_opts(concurrency=2), AppState(), json_mode=False)

    assert sorted(seen) == ["a.mp4", "b.mp4"]
    assert "Captioned 2, skipped 0." in capsys.readouterr().err


def test_batch_emits_ndjson_per_source(monkeypatch, capsys):
    _pipe(monkeypatch, "a.mp4")
    monkeypatch.setattr(caption_exec, "_caption_one", _fake_caption_one([]))

    caption_exec.run_caption(_batch_opts(), AppState(), json_mode=True)

    record = json.loads(capsys.readouterr().out)
    assert record["source"] == "a.mp4"
    assert record["out"] == "a.mp4.captioned"
    assert record["status"] == "completed"


def test_single_source_still_requires_a_media_argument():
    opts = dataclasses.replace(DEFAULTS, media="")
    with pytest.raises(UsageError) as exc:
        caption_exec.run_caption(opts, AppState(), json_mode=False)
    assert "--from-stdin" in (exc.value.suggestion or "") or "--from-stdin" in exc.value.message


# --- skip-on-existing-output (resume) ------------------------------------------


def test_batch_skips_a_source_whose_output_exists(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.captioned.mp4").write_bytes(b"old")  # prior output
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(caption_exec, "_caption_one", _fake_caption_one(seen))

    caption_exec.run_caption(_batch_opts(), AppState(), json_mode=False)

    assert seen == []  # never re-captioned
    assert "Captioned 0, skipped 1." in capsys.readouterr().err


def test_force_recaptions_a_source_whose_output_exists(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.captioned.mp4").write_bytes(b"old")
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(caption_exec, "_caption_one", _fake_caption_one(seen))

    caption_exec.run_caption(_batch_opts(force=True), AppState(), json_mode=False)

    assert seen == [str(src)]  # processed despite the existing output
    assert "Captioned 1, skipped 0." in capsys.readouterr().err


# --- flags that can't span a batch ---------------------------------------------


def test_batch_rejects_out(monkeypatch):
    _pipe(monkeypatch, "a.mp4")
    with pytest.raises(UsageError) as exc:
        caption_exec.run_caption(_batch_opts(out=Path("x.mp4")), AppState(), json_mode=False)
    assert "drop --out" in (exc.value.suggestion or "")


def test_batch_rejects_transcript_id(monkeypatch):
    _pipe(monkeypatch, "a.mp4")
    with pytest.raises(UsageError) as exc:
        caption_exec.run_caption(_batch_opts(transcript_id="tr_1"), AppState(), json_mode=False)
    assert "can't apply to many sources" in (exc.value.suggestion or "")


# --- _existing_output --------------------------------------------------------


def test_existing_output_is_none_for_a_url():
    assert caption_exec._existing_output("https://youtu.be/x") is None


def test_existing_output_is_none_when_missing(tmp_path):
    assert caption_exec._existing_output(str(tmp_path / "a.mp4")) is None


def test_existing_output_returns_the_path_when_present(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "a.captioned.mp4"
    out.write_bytes(b"old")
    assert caption_exec._existing_output(str(src)) == out
