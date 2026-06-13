"""Batch-mode source selection for ``assembly transcribe``.

Splitting a transcribe invocation into its source list — a directory scan, a
glob, a ``--from-stdin`` list, or a bucket URL that is itself a glob/folder — is
a self-contained concern with no dependency on the batch *run* (sidecar resume,
concurrency, output), so it lives here. ``transcribe_batch`` imports the
constants and ``expand_sources``/``reject_single_source_flags`` it needs; the run
machinery stays there. ``transcribe_exec`` calls these directly to decide between
the single-source and batch paths.
"""

from __future__ import annotations

from pathlib import Path

from aai_cli import remotefs, stdio
from aai_cli.errors import UsageError, mutually_exclusive

SIDECAR_SUFFIX = ".aai.json"

# What a directory scan picks up (an explicit glob or stdin list is taken as-is).
AUDIO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".aac",
        ".aif",
        ".aiff",
        ".amr",
        ".flac",
        ".m4a",
        ".m4b",
        ".mka",
        ".mkv",
        ".mov",
        ".mp2",
        ".mp3",
        ".mp4",
        ".mpga",
        ".oga",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)

URL_PREFIXES = ("http://", "https://")
_GLOB_CHARS = frozenset("*?[")


def expand_sources(source: str | None, *, from_stdin: bool, sample: bool) -> list[str] | None:
    """The batch source list, or ``None`` when this is a single-source invocation.

    Batch mode triggers on ``--from-stdin``, a directory (scanned recursively for
    audio files), a glob pattern that names no existing file, or a bucket URL
    that is a glob or trailing-slash folder. A plain file, URL, ``-`` (audio
    piped on stdin), or ``--sample`` stays on the single-source path.
    """
    if from_stdin:
        return _stdin_sources(source, sample=sample)
    # `not source` (rather than `is None`) also catches the empty string — e.g. an
    # unset shell variable in `assembly transcribe "$FILE"`. `Path("")` is `Path(".")`,
    # so it would otherwise fall into the directory branch and batch-transcribe the
    # whole working directory; instead it stays single-source and fails validation.
    if not source or sample or source == "-" or source.startswith(URL_PREFIXES):
        return None
    if remotefs.is_remote_url(source):
        return _remote_sources(source)
    path = Path(source)
    if path.is_dir():
        return _directory_sources(path)
    if not path.exists() and _GLOB_CHARS.intersection(source):
        return _glob_sources(source)
    return None


def _stdin_sources(source: str | None, *, sample: bool) -> list[str]:
    if source is not None or sample:
        raise UsageError(
            "--from-stdin reads sources from stdin; don't also pass a source or --sample."
        )
    lines = list(dict.fromkeys(stdio.iter_piped_stdin_lines()))  # dedupe, keep order
    if not lines:
        raise UsageError(
            "No sources received on stdin.",
            suggestion="Pipe one path or URL per line, e.g. "
            "find . -name '*.mp3' | assembly transcribe --from-stdin.",
        )
    return lines


def _directory_sources(path: Path) -> list[str]:
    files = sorted(
        str(p) for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        raise UsageError(
            f"No audio files found under {path}.",
            suggestion="Recognized extensions: " + ", ".join(sorted(AUDIO_EXTENSIONS)) + ".",
        )
    return files


def _remote_sources(url: str) -> list[str] | None:
    """Batch sources for a bucket/remote URL, or ``None`` when it's a single file.

    Mirrors the local rules (``_glob_sources``/``_directory_sources``): a glob
    expands to its file matches, a trailing-slash folder to its audio files;
    anything else is downloaded as one file.
    """
    if _GLOB_CHARS.intersection(url):
        return _remote_glob_sources(url)
    if url.endswith("/"):
        return _remote_folder_sources(url)
    return None


def _remote_glob_sources(url: str) -> list[str]:
    """The remote files matching a bucket glob, with sidecars excluded."""
    matches = [u for u in remotefs.glob_files(url) if not u.endswith(SIDECAR_SUFFIX)]
    if not matches:
        raise UsageError(f"No files match {url}.")
    return matches


def _remote_folder_sources(url: str) -> list[str]:
    """The audio files under a trailing-slash bucket folder (recursive)."""
    files = [u for u in remotefs.list_files(url) if Path(u).suffix.lower() in AUDIO_EXTENSIONS]
    if not files:
        raise UsageError(
            f"No audio files found under {url}.",
            suggestion="Recognized extensions: " + ", ".join(sorted(AUDIO_EXTENSIONS)) + ".",
        )
    return files


def _glob_sources(pattern: str) -> list[str]:
    # pathlib globs are always relative, so peel an absolute pattern's anchor off
    # and glob from there ("" anchors at the working directory; Path("") is ".").
    anchor = Path(pattern).anchor
    matches = sorted(
        str(p)
        for p in Path(anchor).glob(pattern.removeprefix(anchor))
        if p.is_file() and not str(p).endswith(SIDECAR_SUFFIX)
    )
    if not matches:
        raise UsageError(f"No files match {pattern}.")
    return matches


def reject_single_source_flags(
    *,
    out: Path | None,
    output_field: object | None,
    show_code: bool,
) -> None:
    """Batch mode writes one sidecar per source; the single-result flags don't apply.

    ``--llm`` is deliberately not here: in batch mode the chain runs per source and
    its steps land in each sidecar.
    """
    mutually_exclusive(
        ("--show-code", show_code),
        ("multiple sources", True),
        suggestion="Pass one file or URL with --show-code.",
    )
    mutually_exclusive(
        ("--out", out),
        ("-o/--output", output_field),
        ("multiple sources", True),
        suggestion=f"Each source gets a '{SIDECAR_SUFFIX}' sidecar with the full result.",
    )
