"""Fetching audio from fsspec-addressable storage (s3://, gs://, az://, sftp://, …).

The AssemblyAI API fetches plain http(s) URLs itself, and `youtube.py` handles
media-page URLs; this module covers bucket/remote-storage URLs neither can read.
fsspec core ships with the CLI, but each protocol's backend (s3fs, gcsfs, adlfs,
…) is an optional install — a missing one surfaces as a clean install hint, not a
traceback. Credentials are the backend's business (e.g. the standard AWS
environment/config for s3://), never AssemblyAI's API key.
"""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from aai_cli.core.errors import CLIError

if TYPE_CHECKING:
    from collections.abc import Generator

# Schemes that never route through fsspec even though it knows them: the API
# fetches web URLs itself, and file:// paths take the ordinary local checks.
_NON_REMOTE_PROTOCOLS = frozenset({"http", "https", "file", "local"})


class _RemoteFS(Protocol):
    """The slice of ``fsspec.AbstractFileSystem`` this module touches."""

    @abstractmethod
    def glob(self, path: str, **kwargs: bool) -> dict[str, dict[str, object]]:
        """Expand a glob; with ``detail=True``, a path -> info mapping."""

    @abstractmethod
    def find(self, path: str) -> list[str]:
        """Every file under ``path``, recursively."""

    @abstractmethod
    def unstrip_protocol(self, name: str) -> str:
        """Re-attach the protocol prefix to a bare path."""

    @abstractmethod
    def get_file(self, rpath: str, lpath: str) -> None:
        """Copy one remote file to a local path."""


def is_remote_url(source: str | None) -> bool:
    """True if `source` is a URL whose scheme names an fsspec filesystem (s3://, gs://, …).

    http(s) is excluded (the API fetches web URLs itself), as is file:// (local
    files go through the ordinary path checks); an unknown scheme is not remote.
    """
    if not source or "://" not in source:
        return False
    protocol = source.partition("://")[0]
    if protocol in _NON_REMOTE_PROTOCOLS:
        return False
    from fsspec.registry import known_implementations

    return protocol in known_implementations


def _filesystem(url: str) -> tuple[_RemoteFS, str]:
    """The (filesystem, bare path) pair for `url`.

    A protocol whose backend isn't installed becomes a clean CLIError; fsspec's
    own ImportError text already names the package to install (e.g. "Install
    s3fs to access S3").
    """
    import fsspec

    try:
        # fsspec ships no py.typed; the declared annotation pins the pair's type.
        pair: tuple[_RemoteFS, str] = fsspec.url_to_fs(url)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except ImportError as exc:
        protocol = url.partition("://")[0]
        raise CLIError(
            f"Reading {protocol}:// URLs needs an extra package.",
            error_type="remote_backend_missing",
            exit_code=2,
            suggestion=f"{exc}.",
        ) from exc
    return pair


@contextmanager
def _remote_errors(url: str) -> Generator[None]:
    """Normalize backend exceptions for one remote operation into clean CLIErrors."""
    try:
        yield
    except FileNotFoundError as exc:
        raise CLIError(
            f"Remote file not found: {url}",
            error_type="file_not_found",
            exit_code=2,
            suggestion="Check the path. A folder needs a trailing '/'; globs (*.mp3) also work.",
        ) from exc
    except CLIError:
        raise
    except Exception as exc:  # backends raise many types; surface one clean CLI error
        reason = " ".join(str(exc).split())
        raise CLIError(
            f"Could not access {url}: {reason}",
            error_type="remote_error",
            exit_code=1,
        ) from exc


def download(url: str, dest_dir: Path) -> Path:
    """Copy the remote file at `url` into `dest_dir` and return its local path."""
    fs, path = _filesystem(url)
    local = dest_dir / PurePosixPath(path).name
    with _remote_errors(url):
        fs.get_file(path, str(local))
    return local


def glob_files(url: str) -> list[str]:
    """The full URLs of the files matching the remote glob `url`, sorted."""
    fs, path = _filesystem(url)
    with _remote_errors(url):
        details = fs.glob(path, detail=True)
    return sorted(
        fs.unstrip_protocol(match) for match, info in details.items() if info.get("type") == "file"
    )


def list_files(url: str) -> list[str]:
    """The full URLs of every file under the remote folder/prefix `url`, recursively, sorted."""
    fs, path = _filesystem(url)
    with _remote_errors(url):
        found = fs.find(path)
    return sorted(fs.unstrip_protocol(match) for match in found)
