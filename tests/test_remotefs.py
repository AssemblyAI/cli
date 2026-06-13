"""aai_cli.core.remotefs: fsspec-backed bucket/remote audio sources.

Driven against fsspec's in-process memory filesystem (the shared ``memory_fs``
fixture) so the tests exercise real fsspec glob/find/download code paths while
pytest-socket stays armed.
"""

from pathlib import Path

import fsspec
import pytest
from fsspec.implementations.memory import MemoryFileSystem
from fsspec.registry import known_implementations

from aai_cli.core import remotefs
from aai_cli.core.errors import CLIError


@pytest.mark.parametrize(
    "url",
    [
        "s3://bucket/key.mp3",
        "gs://bucket/key.wav",
        "az://container/key.m4a",
        "sftp://host/path/key.mp3",
        "memory://calls/a.mp3",
    ],
)
def test_is_remote_url_matches_fsspec_schemes(url):
    assert remotefs.is_remote_url(url)


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/a.mp3",  # the API fetches web URLs itself
        "http://example.com/a.mp3",
        "file:///tmp/a.mp3",  # local files take the ordinary path checks
        "local://tmp/a.mp3",
        "./a.mp3",
        "/abs/a.mp3",
        "C:\\audio\\a.mp3",
        "not-a-known-scheme://x/y.mp3",
        "-",
        "",
        None,
    ],
)
def test_is_remote_url_rejects_web_local_and_unknown_sources(source):
    assert not remotefs.is_remote_url(source)


def test_download_copies_the_remote_file_locally(memory_fs, tmp_path):
    memory_fs.pipe("/calls/a.mp3", b"remote-bytes")
    local = remotefs.download("memory://calls/a.mp3", tmp_path)
    assert local == tmp_path / "a.mp3"  # keeps the remote file's name
    assert local.read_bytes() == b"remote-bytes"


def test_download_missing_remote_file_fails_cleanly(memory_fs, tmp_path):
    with pytest.raises(CLIError) as exc:
        remotefs.download("memory://calls/nope.mp3", tmp_path)
    assert exc.value.error_type == "file_not_found"
    assert exc.value.exit_code == 2
    assert exc.value.message == "Remote file not found: memory://calls/nope.mp3"
    assert "trailing '/'" in (exc.value.suggestion or "")


def test_missing_protocol_backend_surfaces_install_hint(tmp_path):
    # fsspec knows the protocol but its backend package isn't importable; the
    # CLIError must carry fsspec's own install hint, not a traceback.
    fsspec.register_implementation(
        "aaimissing", "no_such_pkg.NoFS", errtxt="Install no-such-pkg to access aaimissing"
    )
    try:
        assert remotefs.is_remote_url("aaimissing://bucket/a.mp3")
        with pytest.raises(CLIError) as exc:
            remotefs.download("aaimissing://bucket/a.mp3", tmp_path)
    finally:
        known_implementations.pop("aaimissing")
    assert exc.value.error_type == "remote_backend_missing"
    assert exc.value.exit_code == 2
    assert exc.value.message == "Reading aaimissing:// URLs needs an extra package."
    assert "Install no-such-pkg" in (exc.value.suggestion or "")


def test_backend_errors_become_one_clean_cli_error(memory_fs, tmp_path, monkeypatch):
    # Not-found is special-cased above; anything else (auth, permissions, …)
    # collapses to a single one-line CLIError, multi-line reprs flattened.
    def _denied(self, rpath, lpath, **kwargs):
        raise PermissionError("access\n  denied")

    monkeypatch.setattr(MemoryFileSystem, "get_file", _denied)
    with pytest.raises(CLIError) as exc:
        remotefs.download("memory://calls/a.mp3", tmp_path)
    assert exc.value.error_type == "remote_error"
    assert exc.value.exit_code == 1
    assert exc.value.message == "Could not access memory://calls/a.mp3: access denied"


def test_cli_errors_from_an_operation_pass_through_unwrapped(memory_fs, tmp_path, monkeypatch):
    inner = CLIError("already clean", error_type="x", exit_code=3)

    def _raise(self, rpath, lpath, **kwargs):
        raise inner

    monkeypatch.setattr(MemoryFileSystem, "get_file", _raise)
    with pytest.raises(CLIError) as exc:
        remotefs.download("memory://calls/a.mp3", tmp_path)
    assert exc.value is inner


def test_glob_files_returns_full_urls_for_files_only(memory_fs):
    memory_fs.pipe("/calls/b.mp3", b"b")
    memory_fs.pipe("/calls/a.mp3", b"a")
    memory_fs.pipe("/calls/sub.mp3/inner.mp3", b"i")  # a directory matching the glob
    memory_fs.pipe("/calls/notes.txt", b"x")
    assert remotefs.glob_files("memory://calls/*.mp3") == [
        "memory:///calls/a.mp3",
        "memory:///calls/b.mp3",
    ]


def test_list_files_walks_the_folder_recursively(memory_fs):
    memory_fs.pipe("/calls/a.mp3", b"a")
    memory_fs.pipe("/calls/sub/b.wav", b"b")
    assert remotefs.list_files("memory://calls/") == [
        "memory:///calls/a.mp3",
        "memory:///calls/sub/b.wav",
    ]


def test_results_are_sorted_even_when_the_backend_is_not(memory_fs, monkeypatch):
    # Real backends return listing order; the expansion must be deterministic.
    monkeypatch.setattr(MemoryFileSystem, "find", lambda self, path: ["/z.mp3", "/a.mp3"])
    assert remotefs.list_files("memory://x/") == ["memory:///a.mp3", "memory:///z.mp3"]
    monkeypatch.setattr(
        MemoryFileSystem,
        "glob",
        lambda self, path, **kwargs: {"/z.mp3": {"type": "file"}, "/a.mp3": {"type": "file"}},
    )
    assert remotefs.glob_files("memory://x/*.mp3") == ["memory:///a.mp3", "memory:///z.mp3"]


def test_downloaded_urls_round_trip_through_url_to_fs(memory_fs, tmp_path):
    # glob_files/list_files emit unstripped URLs (memory:///…); download must
    # accept exactly those, since batch mode feeds them straight back in.
    memory_fs.pipe("/calls/a.mp3", b"abc")
    (url,) = remotefs.glob_files("memory://calls/*.mp3")
    local = remotefs.download(url, tmp_path)
    assert Path(local).read_bytes() == b"abc"
