"""Batch-mode runs for `assembly transcribe`: sidecar resume, concurrency, failures,
and the per-source NDJSON / progress-table output.

Source selection (glob/directory/stdin expansion, rejected flags) lives in
test_transcribe_batch_sources.py.
"""

import hashlib
import json

import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import batch as transcribe_batch
from aai_cli.core import config
from aai_cli.core.errors import auth_failure
from aai_cli.main import app

runner = CliRunner()

_TRANSCRIBE = "aai_cli.app.transcribe.run.client.transcribe"


@pytest.fixture(autouse=True)
def workdir(tmp_path, monkeypatch):
    # Batch sources and sidecars are resolved relative to the working directory;
    # isolate each test in its own tmp cwd.
    monkeypatch.chdir(tmp_path)


def _auth():
    config.set_api_key("default", "sk_live")


def _fake_transcript(mocker, source="x"):
    t = mocker.MagicMock()
    t.id = f"t_{source}"
    t.text = f"text of {source}"
    t.status = "completed"
    t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
    return t


def _patch_transcribe(mocker, monkeypatch):
    """Patch client.transcribe with a fake that records the audio args it saw."""
    seen = []

    def fake(api_key, audio, *, config):
        seen.append(audio)
        return _fake_transcript(mocker, audio)

    monkeypatch.setattr(_TRANSCRIBE, fake)
    return seen


def _ndjson(result):
    return [json.loads(line) for line in result.output.splitlines() if line.startswith("{")]


def test_sidecar_file_is_two_space_indented_with_trailing_newline(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _patch_transcribe(mocker, monkeypatch)
    runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    text = (tmp_path / "a.mp3.aai.json").read_text()
    assert text.startswith('{\n  "source"')  # indent=2
    assert text.endswith("}\n")


# --- sidecar resume -------------------------------------------------------------


def _completed_sidecar(tmp_path, name, data, transcript_id="t_old"):
    record = {
        "source": name,
        "id": transcript_id,
        "status": "completed",
        "transcript": {"id": transcript_id},
        "source_sha256": hashlib.sha256(data).hexdigest(),
    }
    (tmp_path / f"{name}.aai.json").write_text(json.dumps(record))


def test_rerun_skips_sources_with_completed_sidecars(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "b.mp3").write_bytes(b"bbb")
    _completed_sidecar(tmp_path, "a.mp3", b"aaa")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen == ["b.mp3"]  # only the unfinished source pays
    records = {r["source"]: r for r in _ndjson(result)}
    assert records["a.mp3"] == {
        "type": "result",
        "source": "a.mp3",
        "status": "skipped",
        "id": "t_old",
        "sidecar": "a.mp3.aai.json",
    }


def test_changed_file_bytes_invalidate_the_sidecar(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"new-bytes")
    _completed_sidecar(tmp_path, "a.mp3", b"old-bytes")  # hash no longer matches
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]


def test_force_retranscribes_despite_completed_sidecar(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _completed_sidecar(tmp_path, "a.mp3", b"aaa")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--force", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]


def test_corrupt_sidecar_retranscribes(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "a.mp3.aai.json").write_text("not json{")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]


def test_non_completed_sidecar_retranscribes(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "a.mp3.aai.json").write_text(json.dumps({"status": "error"}))
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]


def test_non_dict_sidecar_retranscribes(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "a.mp3.aai.json").write_text(json.dumps(["completed"]))
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]


def test_url_source_resumes_on_sidecar_alone(tmp_path, mocker, monkeypatch):
    # URLs have no local bytes to hash: a completed sidecar is the whole resume check.
    _auth()
    url = "https://example.com/ep.mp3"
    sidecar = transcribe_batch.sidecar_path(url)
    sidecar.write_text(json.dumps({"status": "completed", "id": "t_old"}))
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "--from-stdin", "--json"], input=url + "\n")
    assert result.exit_code == 0
    assert seen == []
    assert _ndjson(result) == [
        {
            "type": "result",
            "source": url,
            "status": "skipped",
            "id": "t_old",
            "sidecar": str(sidecar),
        }
    ]


def test_url_sidecar_omits_source_hash(tmp_path, mocker, monkeypatch):
    _auth()
    url = "https://example.com/ep.mp3"
    _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "--from-stdin", "--json"], input=url + "\n")
    assert result.exit_code == 0
    record = json.loads(transcribe_batch.sidecar_path(url).read_text())
    assert record["source"] == url
    assert "source_sha256" not in record


def test_url_sidecar_name_is_a_slug_plus_url_hash():
    url = "https://example.com/shows/ep.mp3?token=a/b"
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    assert (
        str(transcribe_batch.sidecar_path(url))
        == f"example.com-shows-ep.mp3-token-a-b-{digest}.aai.json"
    )


def test_url_sidecar_slug_truncates_to_64_chars():
    url = "https://example.com/" + "x" * 100
    name = transcribe_batch.sidecar_path(url).name
    slug = name.removesuffix(".aai.json").rsplit("-", 1)[0]
    assert slug == ("example.com/" + "x" * 100).replace("/", "-")[:64]
    assert len(slug) == 64


# --- failures, exit codes, auth --------------------------------------------------


def test_partial_failure_exits_1_and_completes_the_rest(tmp_path, mocker, monkeypatch):
    from aai_cli.core.errors import APIError

    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    (tmp_path / "b.mp3").write_bytes(b"b")

    def fake(api_key, audio, *, config):
        if audio == "a.mp3":
            raise APIError("upload exploded")
        return _fake_transcript(mocker, audio)

    monkeypatch.setattr(_TRANSCRIBE, fake)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 1
    records = {r["source"]: r for r in _ndjson(result) if "source" in r}
    assert records["a.mp3"] == {
        "type": "result",
        "source": "a.mp3",
        "status": "failed",
        "error": "upload exploded",
    }
    assert records["b.mp3"]["status"] == "completed"
    assert (tmp_path / "b.mp3.aai.json").exists()
    assert not (tmp_path / "a.mp3.aai.json").exists()
    assert "1 of 2 sources failed." in result.output
    assert "Re-run the same command" in result.output  # resume is the retry path


def test_rejected_key_aborts_the_batch_with_exit_4(tmp_path, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")

    def fake(api_key, audio, *, config):
        raise auth_failure()

    monkeypatch.setattr(_TRANSCRIBE, fake)
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: False)
    result = runner.invoke(app, ["transcribe", "*.mp3"])
    assert result.exit_code == 4
    assert "rejected" in result.output


def test_auth_failure_drops_not_yet_started_sources(tmp_path, monkeypatch):
    # The abort path must shut the pool down with cancel_futures=True — that's what
    # keeps one rejected key from burning an upload per queued source. Asserted on
    # the shutdown call because which queued futures actually get dropped is a race.
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    seen = {}
    real = transcribe_batch.ThreadPoolExecutor

    class Capture(real):
        def shutdown(self, wait=True, *, cancel_futures=False):
            seen.setdefault("cancel_futures", cancel_futures)  # first call wins
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(transcribe_batch, "ThreadPoolExecutor", Capture)

    def fake(api_key, audio, *, config):
        raise auth_failure()

    monkeypatch.setattr(_TRANSCRIBE, fake)
    monkeypatch.setattr("aai_cli.app.context._interactive_session", lambda: False)
    result = runner.invoke(app, ["transcribe", "*.mp3"])
    assert result.exit_code == 4
    assert seen["cancel_futures"] is True


# --- output rendering -------------------------------------------------------------


def test_human_mode_prints_result_table_and_summary(tmp_path, mocker, monkeypatch):
    # 2 transcribed + 1 skipped: asymmetric on purpose, so a summary that counted
    # the *non*-completed sources would read "Transcribed 1, skipped 2" and fail.
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "b.mp3").write_bytes(b"bbb")
    (tmp_path / "c.mp3").write_bytes(b"ccc")
    _completed_sidecar(tmp_path, "c.mp3", b"ccc")
    _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3"])
    assert result.exit_code == 0
    assert "Source" in result.output and "Status" in result.output
    assert "completed" in result.output and "skipped" in result.output
    assert "a.mp3.aai.json" in result.output
    assert "Transcribed 2, skipped 1." in result.output
    assert "{" not in result.output  # human mode emits no NDJSON


def test_json_mode_emits_no_table_or_summary(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert "Source" not in result.output  # no table
    assert "Transcribed" not in result.output  # no human summary


def test_quiet_suppresses_the_summary_but_not_the_table(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["--quiet", "transcribe", "*.mp3"])
    assert result.exit_code == 0
    assert "Source" in result.output  # the table is the data
    assert "Transcribed" not in result.output


# --- concurrency ------------------------------------------------------------------


def _capture_pool_size(monkeypatch):
    seen = {}
    real = transcribe_batch.ThreadPoolExecutor

    class Capture(real):
        def __init__(self, max_workers=None, **kwargs):
            seen["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(transcribe_batch, "ThreadPoolExecutor", Capture)
    return seen


def test_default_concurrency_is_four(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    _patch_transcribe(mocker, monkeypatch)
    seen = _capture_pool_size(monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert seen["max_workers"] == 4


def test_concurrency_flag_sets_pool_size(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    _patch_transcribe(mocker, monkeypatch)
    seen = _capture_pool_size(monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--concurrency", "1", "--json"])
    assert result.exit_code == 0
    assert seen["max_workers"] == 1


def test_concurrency_below_one_is_rejected(tmp_path):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    result = runner.invoke(app, ["transcribe", "*.mp3", "--concurrency", "0"])
    assert result.exit_code == 2


def test_batch_runs_sources_concurrently(tmp_path, mocker, monkeypatch):
    # With concurrency 2 and two sources, both workers must be in flight at once.
    import threading

    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    (tmp_path / "b.mp3").write_bytes(b"b")
    barrier = threading.Barrier(2, timeout=10)

    def fake(api_key, audio, *, config):
        barrier.wait()  # deadlocks (and times out) unless both run concurrently
        return _fake_transcript(mocker, audio)

    monkeypatch.setattr(_TRANSCRIBE, fake)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--concurrency", "2", "--json"])
    assert result.exit_code == 0
    assert len(_ndjson(result)) == 2


# --- unit edges --------------------------------------------------------------------


def test_resumable_record_requires_matching_hash():
    # Direct unit check of the three local-file outcomes: match, mismatch, missing.
    import pathlib

    sidecar = pathlib.Path("side.aai.json")
    sidecar.write_text(json.dumps({"status": "completed", "source_sha256": "abc"}))
    assert transcribe_batch.resumable_record(sidecar, digest="abc") is not None
    assert transcribe_batch.resumable_record(sidecar, digest="other") is None
    assert transcribe_batch.resumable_record(pathlib.Path("missing.json"), digest="abc") is None


def test_item_record_minimal_shape_before_any_work():
    # A not-yet-finished item serializes to just {source, status}: no empty id/sidecar keys.
    assert transcribe_batch._Item("x.mp3").record() == {
        "type": "result",
        "source": "x.mp3",
        "status": "queued",
    }


def test_source_digest_is_sha256_of_file_bytes(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"payload")
    assert transcribe_batch._source_digest("a.mp3") == hashlib.sha256(b"payload").hexdigest()
    assert transcribe_batch._source_digest("https://example.com/a.mp3") is None
    assert transcribe_batch._source_digest("missing.mp3") is None
