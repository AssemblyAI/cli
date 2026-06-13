"""Batch-mode source selection for `assembly transcribe`: glob/directory/stdin
expansion (local and remote/bucket URLs) and the single-source flags batch mode
rejects.

The batch *run* (sidecar resume, concurrency, failures, output) lives in
test_transcribe_batch.py. Remote sources use fsspec's memory:// filesystem (the
shared ``memory_fs`` fixture), so the real download/glob paths run offline.
"""

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aai_cli.app.transcribe import batch as transcribe_batch
from aai_cli.app.transcribe import sources as transcribe_sources
from aai_cli.core import config
from aai_cli.core.errors import UsageError
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


def _patch_transcribe(mocker, monkeypatch):
    """Patch client.transcribe with a fake that records the audio args it saw."""
    seen = []

    def fake(api_key, audio, *, config):
        seen.append(audio)
        t = mocker.MagicMock()
        t.id = f"t_{audio}"
        t.text = f"text of {audio}"
        t.status = "completed"
        t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
        return t

    monkeypatch.setattr(_TRANSCRIBE, fake)
    return seen


def test_glob_skips_sidecar_files(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "stale.aai.json").write_text("{}")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*", "--json"])
    assert result.exit_code == 0
    assert seen == ["a.mp3"]  # the stray sidecar is never treated as audio


def test_absolute_glob_pattern_matches_from_its_anchor(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", str(tmp_path / "*.mp3"), "--json"])
    assert result.exit_code == 0
    assert seen == [str(tmp_path / "a.mp3")]


def test_glob_without_matches_exits_2(mocker, monkeypatch):
    _auth()
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "missing-*.mp3"])
    assert result.exit_code == 2
    assert "No files match" in result.output
    assert seen == []


def test_existing_file_with_glob_chars_stays_single_source(tmp_path, mocker, monkeypatch):
    # A real file whose name contains [ ] must not be re-interpreted as a pattern.
    _auth()
    (tmp_path / "take[1].mp3").write_bytes(b"aaa")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "take[1].mp3", "-o", "id"])
    assert result.exit_code == 0
    assert seen == ["take[1].mp3"]
    assert result.output.strip() == "t_take[1].mp3"  # single-source output, no sidecar table


def test_directory_scan_is_recursive_and_audio_only(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "calls").mkdir()
    (tmp_path / "calls" / "sub").mkdir()
    (tmp_path / "calls" / "a.mp3").write_bytes(b"a")
    (tmp_path / "calls" / "sub" / "b.WAV").write_bytes(b"b")  # extension match is case-insensitive
    (tmp_path / "calls" / "notes.txt").write_text("not audio")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "calls", "--json"])
    assert result.exit_code == 0
    assert sorted(seen) == ["calls/a.mp3", "calls/sub/b.WAV"]


def test_directory_without_audio_exits_2(tmp_path):
    _auth()
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("x")
    result = runner.invoke(app, ["transcribe", "empty"])
    assert result.exit_code == 2
    assert "No audio files found" in result.output
    assert ".mp3" in result.output  # the suggestion lists recognized extensions


def test_from_stdin_reads_deduped_lines(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    (tmp_path / "b.mp3").write_bytes(b"b")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(
        app, ["transcribe", "--from-stdin", "--json"], input="a.mp3\n\na.mp3\nb.mp3\n"
    )
    assert result.exit_code == 0
    assert sorted(seen) == ["a.mp3", "b.mp3"]  # blank line dropped, duplicate collapsed


def test_stdin_source_list_dedupes_preserving_order(monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("b.mp3\na.mp3\nb.mp3\n"))
    assert transcribe_sources.expand_sources(None, from_stdin=True, sample=False) == [
        "b.mp3",
        "a.mp3",
    ]


def test_from_stdin_with_empty_stdin_exits_2():
    _auth()
    result = runner.invoke(app, ["transcribe", "--from-stdin"], input="")
    assert result.exit_code == 2
    assert "No sources received on stdin" in result.output


def test_from_stdin_rejects_source_argument():
    _auth()
    result = runner.invoke(app, ["transcribe", "a.mp3", "--from-stdin"], input="b.mp3\n")
    assert result.exit_code == 2
    assert "--from-stdin reads sources from stdin" in result.output


def test_from_stdin_rejects_sample():
    _auth()
    result = runner.invoke(app, ["transcribe", "--sample", "--from-stdin"], input="b.mp3\n")
    assert result.exit_code == 2
    assert "--from-stdin reads sources from stdin" in result.output


@pytest.mark.parametrize("source", ["-", "https://example.com/a.mp3", None, ""])
def test_non_batch_sources_return_none(source):
    assert transcribe_sources.expand_sources(source, from_stdin=False, sample=False) is None


def test_empty_source_is_rejected_not_treated_as_cwd(tmp_path, mocker, monkeypatch):
    # Path("") == Path("."), so an empty source (e.g. an unset shell variable in
    # `assembly transcribe "$FILE"`) used to batch-transcribe the whole working
    # directory; it must fail like a missing source instead.
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", ""])
    assert result.exit_code == 2
    assert "Provide an audio path or URL." in result.output
    assert seen == []  # nothing in the cwd was picked up, let alone transcribed


def test_sample_returns_none_even_without_source():
    assert transcribe_sources.expand_sources(None, from_stdin=False, sample=True) is None


def test_expand_sources_directory_error_message_names_the_path(tmp_path):
    (tmp_path / "calls").mkdir()
    with pytest.raises(UsageError, match="No audio files found under calls"):
        transcribe_sources.expand_sources("calls", from_stdin=False, sample=False)


@pytest.mark.parametrize(
    ("extra", "flag_name", "hint"),
    [
        (["--out", "x.txt"], "--out", "sidecar with the full result"),
        (["-o", "text"], "-o/--output", "sidecar with the full result"),
        (["--show-code"], "--show-code", "Pass one file or URL"),
    ],
)
def test_batch_rejects_single_source_flags(tmp_path, extra, flag_name, hint):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"a")
    result = runner.invoke(app, ["transcribe", "*.mp3", *extra])
    assert result.exit_code == 2
    assert f"{flag_name} and multiple sources can't be combined." in result.output
    assert hint in result.output


def _patch_transcribe_recording_uploads(mocker, monkeypatch):
    """Patch client.transcribe with a fake that records (name, bytes, path) per upload.

    Remote sources are downloaded to a temp file before upload, so the audio arg
    is a local path whose basename and bytes must match the remote object.
    """
    seen = []

    def fake(api_key, audio, *, config):
        seen.append((Path(audio).name, Path(audio).read_bytes(), audio))
        t = mocker.MagicMock()
        t.id = f"t_{Path(audio).name}"
        t.text = "hi"
        t.status = "completed"
        t.json_response = {"id": t.id, "text": t.text, "status": "completed"}
        return t

    monkeypatch.setattr(_TRANSCRIBE, fake)
    return seen


def test_remote_glob_batch_downloads_and_transcribes(mocker, monkeypatch, memory_fs):
    _auth()
    memory_fs.pipe("/calls/a.mp3", b"aaa")
    memory_fs.pipe("/calls/b.mp3", b"bbb")
    memory_fs.pipe("/calls/notes.txt", b"not audio")
    seen = _patch_transcribe_recording_uploads(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "memory://calls/*.mp3", "--json"])
    assert result.exit_code == 0
    # Each match was fetched to a temp dir (never the cwd) and its bytes uploaded.
    assert sorted((name, data) for name, data, _ in seen) == [
        ("a.mp3", b"aaa"),
        ("b.mp3", b"bbb"),
    ]
    assert all("aai-remote-" in path for _, _, path in seen)
    records = {r["source"]: r for r in map(json.loads, result.output.splitlines())}
    record = records["memory:///calls/a.mp3"]
    assert record["status"] == "completed"
    digest = hashlib.sha256(b"memory:///calls/a.mp3").hexdigest()[:8]
    assert record["sidecar"] == f"calls-a.mp3-{digest}.aai.json"
    assert json.loads(Path(record["sidecar"]).read_text())["status"] == "completed"


def test_remote_batch_rerun_resumes_from_sidecars(mocker, monkeypatch, memory_fs):
    _auth()
    memory_fs.pipe("/calls/a.mp3", b"aaa")
    seen = _patch_transcribe_recording_uploads(mocker, monkeypatch)
    assert runner.invoke(app, ["transcribe", "memory://calls/*.mp3", "--json"]).exit_code == 0
    result = runner.invoke(app, ["transcribe", "memory://calls/*.mp3", "--json"])
    assert result.exit_code == 0
    assert [json.loads(line)["status"] for line in result.output.splitlines()] == ["skipped"]
    assert len(seen) == 1  # the re-run never downloaded or transcribed again


def test_remote_glob_skips_sidecar_files(mocker, monkeypatch, memory_fs):
    _auth()
    memory_fs.pipe("/calls/a.mp3", b"aaa")
    memory_fs.pipe("/calls/stale.aai.json", b"{}")
    seen = _patch_transcribe_recording_uploads(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "memory://calls/*", "--json"])
    assert result.exit_code == 0
    assert [name for name, _, _ in seen] == ["a.mp3"]


def test_remote_folder_scan_is_recursive_and_audio_only(mocker, monkeypatch, memory_fs):
    _auth()
    memory_fs.pipe("/calls/a.mp3", b"a")
    memory_fs.pipe("/calls/sub/b.WAV", b"b")  # extension match is case-insensitive
    memory_fs.pipe("/calls/notes.txt", b"not audio")
    seen = _patch_transcribe_recording_uploads(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "memory://calls/", "--json"])
    assert result.exit_code == 0
    assert sorted(name for name, _, _ in seen) == ["a.mp3", "b.WAV"]


def test_remote_folder_without_audio_exits_2(memory_fs):
    _auth()
    memory_fs.pipe("/calls/notes.txt", b"x")
    result = runner.invoke(app, ["transcribe", "memory://calls/"])
    assert result.exit_code == 2
    assert "No audio files found under memory://calls/" in result.output
    assert ".mp3" in result.output  # the suggestion lists recognized extensions


def test_remote_glob_without_matches_exits_2(memory_fs):
    _auth()
    result = runner.invoke(app, ["transcribe", "memory://calls/*.mp3"])
    assert result.exit_code == 2
    assert "No files match" in result.output


def test_plain_remote_file_url_stays_single_source(memory_fs):
    # No glob and no trailing slash: a bucket URL is one file, like a local path.
    for url in ("memory://calls/a.mp3", "memory://calls"):
        assert transcribe_sources.expand_sources(url, from_stdin=False, sample=False) is None


def test_sidecar_path_for_remote_url_is_slug_plus_hash():
    url = "memory:///calls/a.mp3"
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    assert transcribe_batch.sidecar_path(url) == Path(f"calls-a.mp3-{digest}.aai.json")


def test_glob_batch_writes_per_source_sidecars(tmp_path, mocker, monkeypatch):
    _auth()
    (tmp_path / "a.mp3").write_bytes(b"aaa")
    (tmp_path / "b.mp3").write_bytes(b"bbb")
    seen = _patch_transcribe(mocker, monkeypatch)
    result = runner.invoke(app, ["transcribe", "*.mp3", "--json"])
    assert result.exit_code == 0
    assert sorted(seen) == ["a.mp3", "b.mp3"]
    records = {r["source"]: r for r in map(json.loads, result.output.splitlines())}
    assert records["a.mp3"] == {
        "type": "result",
        "source": "a.mp3",
        "status": "completed",
        "id": "t_a.mp3",
        "sidecar": "a.mp3.aai.json",
    }
    sidecar = json.loads((tmp_path / "a.mp3.aai.json").read_text())
    assert sidecar["status"] == "completed"
    assert sidecar["transcript"] == {
        "id": "t_a.mp3",
        "text": "text of a.mp3",
        "status": "completed",
    }
    assert sidecar["source_sha256"] == hashlib.sha256(b"aaa").hexdigest()
