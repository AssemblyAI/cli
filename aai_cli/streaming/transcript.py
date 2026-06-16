"""Write finalized turn text to a file — backs `assembly stream --save-transcript`/`--save-dir`.

Parallels ``record.tee_wav`` (which keeps the raw PCM): this keeps the human-readable
transcript, one finalized turn per line — exactly the plain text ``-o text`` emits —
flushed as each turn lands so a Ctrl-C still leaves the transcript captured so far.
"""

from __future__ import annotations

from pathlib import Path

from aai_cli.core.errors import CLIError


def validate_target(path: Path) -> None:
    """Reject a transcript path whose parent directory is missing, before streaming.

    Run before credentials/audio are opened so a bad path reads as a path error up
    front, not after a session has already started writing into the void.
    """
    parent = path.parent
    if not parent.is_dir():
        raise CLIError(
            f"Cannot save transcript to {path}: {parent} is not a directory.",
            error_type="save_transcript_path",
            exit_code=2,
            suggestion="Create the directory first, or pass a path under an existing one.",
        )


class TranscriptWriter:
    """Append finalized turn lines to a UTF-8 text file, flushing each one.

    Flushing per turn means a mid-stream stop (Ctrl-C) still leaves a complete
    transcript of the turns seen so far, the same robustness ``record.tee_wav`` gives
    the audio.
    """

    def __init__(self, path: Path) -> None:
        try:
            # Open the handle up front so a bad path fails here cleanly, mirroring
            # record.tee_wav rather than discovering the error on the first turn.
            self._handle = path.open("w", encoding="utf-8")
        except OSError as exc:
            raise CLIError(
                f"Cannot open {path} for writing: {exc}",
                error_type="save_transcript_path",
                exit_code=2,
            ) from exc

    def write_turn(self, line: str) -> None:
        """Write one finalized turn as its own line and flush it to disk immediately."""
        self._handle.write(f"{line}\n")
        self._handle.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._handle.close()
