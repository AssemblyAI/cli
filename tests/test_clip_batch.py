"""Batch-mode tests for `assembly clip` (--from-stdin / --concurrency / --force).

The single-source pipeline (`_clip_one` → transcript selection → ffmpeg cuts) is
covered in test_clip_exec.py; here it is faked so the tests pin only the batch
wiring: source dispatch, the skip-when-clips-already-exist worker, and the
transcript-id flag that can't span a batch. clip resolves ffmpeg and validates
the (shared) selection before dispatch, so those run for real.
"""

from __future__ import annotations

import dataclasses

import pytest

from aai_cli.app.context import AppState
from aai_cli.commands.clip import _exec as clip_exec
from aai_cli.core import stdio
from aai_cli.core.errors import UsageError
from tests._clip_helpers import DEFAULTS


@pytest.fixture(autouse=True)
def _ffmpeg_on_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")


def _pipe(monkeypatch: pytest.MonkeyPatch, *sources: str) -> None:
    monkeypatch.setattr(stdio, "iter_piped_stdin_lines", lambda: iter(sources))


def _fake_clip_one(seen: list[str], *, clips: int = 2):
    def one(opts, explicit, ffmpeg, state, *, json_mode):
        # The batch worker silences spinners by handing each source a quiet state.
        assert state.quiet is True
        seen.append(opts.media)
        payload = {"source": opts.media, "transcript_id": "tr", "clips": [{}] * clips}
        return payload, [object()] * clips

    return one


def _batch_opts(**overrides):
    # A selector is required (validated before dispatch); --speaker applies per source.
    return dataclasses.replace(DEFAULTS, media="", from_stdin=True, speakers=["A"], **overrides)


def test_batch_clips_each_piped_source(monkeypatch, capsys):
    _pipe(monkeypatch, "a.mp4", "b.mp4")
    seen: list[str] = []
    monkeypatch.setattr(clip_exec, "_clip_one", _fake_clip_one(seen))

    clip_exec.run_clip(_batch_opts(concurrency=2), AppState(), json_mode=False)

    assert sorted(seen) == ["a.mp4", "b.mp4"]
    captured = capsys.readouterr()
    assert "Clipped 2, skipped 0." in captured.err
    assert "2 clip(s)" in captured.out  # the per-source summary in the table


def test_single_source_still_requires_a_media_argument():
    opts = dataclasses.replace(DEFAULTS, media="", speakers=["A"])
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(opts, AppState(), json_mode=False)
    assert "--from-stdin" in (exc.value.suggestion or "") or "--from-stdin" in exc.value.message


def test_batch_skips_a_source_that_already_has_clips(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.clip01.mp4").write_bytes(b"old")  # a prior clip
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(clip_exec, "_clip_one", _fake_clip_one(seen))

    clip_exec.run_clip(_batch_opts(), AppState(), json_mode=False)

    assert seen == []
    assert "Clipped 0, skipped 1." in capsys.readouterr().err


def test_force_reclips_a_source_that_already_has_clips(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.clip01.mp4").write_bytes(b"old")
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(clip_exec, "_clip_one", _fake_clip_one(seen))

    clip_exec.run_clip(_batch_opts(force=True), AppState(), json_mode=False)

    assert seen == [str(src)]
    assert "Clipped 1, skipped 0." in capsys.readouterr().err


def test_batch_rejects_transcript_id(monkeypatch):
    _pipe(monkeypatch, "a.mp4")
    with pytest.raises(UsageError) as exc:
        clip_exec.run_clip(_batch_opts(transcript_id="tr_1"), AppState(), json_mode=False)
    assert "can't apply to many sources" in (exc.value.suggestion or "")


# --- _clips_exist ------------------------------------------------------------


def test_clips_exist_is_false_for_a_url():
    assert clip_exec._clips_exist("https://youtu.be/x", None) is False


def test_clips_exist_is_false_when_no_clip_files(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    assert clip_exec._clips_exist(str(src), None) is False


def test_clips_exist_is_true_when_a_clip_file_is_present(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.clip03.mp4").write_bytes(b"old")
    assert clip_exec._clips_exist(str(src), None) is True


def test_clips_exist_checks_the_out_dir_when_given(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    out_dir = tmp_path / "clips"
    out_dir.mkdir()
    # The clip would land in out_dir, not next to the source.
    assert clip_exec._clips_exist(str(src), out_dir) is False
    (out_dir / "a.clip01.mp4").write_bytes(b"old")
    assert clip_exec._clips_exist(str(src), out_dir) is True
