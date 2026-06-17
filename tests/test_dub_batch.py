"""Batch-mode tests for `assembly dub` (--from-stdin / --concurrency / --force).

The single-source pipeline (`_dub_one`) is covered in test_dub_exec.py; here it
is faked so the tests pin only the batch wiring: source dispatch, the
skip-on-existing-output worker, and the single-result flags that can't span a
batch. The invocation-global setup dub runs before dispatch (sandbox/session,
--voice parsing) still executes, so the sandbox fixture is required.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from aai_cli.app import batch
from aai_cli.app.context import AppState
from aai_cli.commands.dub import _exec as dub_exec
from aai_cli.core import stdio
from aai_cli.core.errors import UsageError
from tests._dub_helpers import DEFAULTS, enable_sandbox, patch_api_key


@pytest.fixture(autouse=True)
def _sandbox_and_key(monkeypatch: pytest.MonkeyPatch):
    enable_sandbox(monkeypatch)
    patch_api_key(monkeypatch)


def _pipe(monkeypatch: pytest.MonkeyPatch, *sources: str) -> None:
    monkeypatch.setattr(stdio, "iter_piped_stdin_lines", lambda: iter(sources))


def _fake_dub_one(seen: list[str]):
    def one(opts, state, language, voice_plan, *, json_mode):
        # The batch worker silences spinners by handing each source a quiet state.
        assert state.quiet is True
        seen.append(opts.media)
        return batch.SourceResult(
            payload={"source": opts.media, "out": f"{opts.media}.dub.de"},
            summary=f"{opts.media} dubbed",
        )

    return one


def _batch_opts(**overrides):
    return dataclasses.replace(DEFAULTS, media="", from_stdin=True, **overrides)


def test_batch_dubs_each_piped_source(monkeypatch, capsys):
    _pipe(monkeypatch, "a.mp4", "b.mp4")
    seen: list[str] = []
    monkeypatch.setattr(dub_exec, "_dub_one", _fake_dub_one(seen))

    dub_exec.run_dub(_batch_opts(concurrency=2), AppState(), json_mode=False)

    assert sorted(seen) == ["a.mp4", "b.mp4"]
    assert "Dubbed 2, skipped 0." in capsys.readouterr().err


def test_single_source_still_requires_a_media_argument(monkeypatch):
    opts = dataclasses.replace(DEFAULTS, media="")
    with pytest.raises(UsageError) as exc:
        dub_exec.run_dub(opts, AppState(), json_mode=False)
    assert "--from-stdin" in (exc.value.suggestion or "") or "--from-stdin" in exc.value.message


def test_batch_skips_a_source_whose_output_exists(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.dub.german.mp4").write_bytes(b"old")  # prior output for --lang de
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(dub_exec, "_dub_one", _fake_dub_one(seen))

    dub_exec.run_dub(_batch_opts(), AppState(), json_mode=False)

    assert seen == []
    assert "Dubbed 0, skipped 1." in capsys.readouterr().err


def test_force_redubs_a_source_whose_output_exists(tmp_path, monkeypatch, capsys):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    (tmp_path / "a.dub.german.mp4").write_bytes(b"old")
    _pipe(monkeypatch, str(src))
    seen: list[str] = []
    monkeypatch.setattr(dub_exec, "_dub_one", _fake_dub_one(seen))

    dub_exec.run_dub(_batch_opts(force=True), AppState(), json_mode=False)

    assert seen == [str(src)]
    assert "Dubbed 1, skipped 0." in capsys.readouterr().err


def test_batch_rejects_out(monkeypatch):
    _pipe(monkeypatch, "a.mp4")
    with pytest.raises(UsageError) as exc:
        dub_exec.run_dub(_batch_opts(out=Path("x.mp4")), AppState(), json_mode=False)
    assert "drop --out" in (exc.value.suggestion or "")


def test_batch_rejects_transcript_id(monkeypatch):
    _pipe(monkeypatch, "a.mp4")
    with pytest.raises(UsageError) as exc:
        dub_exec.run_dub(_batch_opts(transcript_id="tr_1"), AppState(), json_mode=False)
    assert "can't apply to many sources" in (exc.value.suggestion or "")


# --- _existing_output --------------------------------------------------------


def test_existing_output_is_none_for_a_url():
    assert dub_exec._existing_output("https://youtu.be/x", "German") is None


def test_existing_output_is_none_when_missing(tmp_path):
    assert dub_exec._existing_output(str(tmp_path / "a.mp4"), "German") is None


def test_existing_output_returns_the_path_when_present(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "a.dub.german.mp4"
    out.write_bytes(b"old")
    assert dub_exec._existing_output(str(src), "German") == out
