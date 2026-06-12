"""Shared scaffolding for commands that operate on a local media file (clip,
dub): source validation, ffmpeg discovery/invocation, and resolution of the
diarized transcript that drives selection or dubbing.

The helpers raise identical CLIErrors regardless of the calling command — only
the command-name/purpose strings differ — so the media-file UX of `assembly
clip` and `assembly dub` can't drift apart.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import assemblyai as aai

from aai_cli import client, output
from aai_cli.errors import CLIError


def validate_local_media(media: Path, command: str) -> None:
    """Reject a missing local source before credential resolution, so a typo'd
    path reads as "file not found", never as a login prompt or an opaque
    ffmpeg error."""
    if not media.exists():
        raise CLIError(
            f"File not found: {media}",
            error_type="file_not_found",
            exit_code=2,
            suggestion=f"Check the path. assembly {command} needs a local audio/video file.",
        )
    if not media.is_file():
        raise CLIError(
            f"Not a file: {media}",
            error_type="not_a_file",
            exit_code=2,
            suggestion="Pass a media file, not a directory.",
        )


def require_ffmpeg(purpose: str) -> str:
    """The ffmpeg executable; checked before any (billed) transcription work."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise CLIError(
            f"ffmpeg is required to {purpose}, but it isn't on PATH.",
            error_type="missing_dependency",
            suggestion="Install it (brew install ffmpeg / apt install ffmpeg) and re-run.",
        )
    return path


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Boundary seam for tests: one ffmpeg invocation, output captured."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def ffmpeg_failure(
    result: subprocess.CompletedProcess[str],
    action: str,
    dest: Path,
    *,
    error_type: str,
) -> CLIError:
    """A failed ffmpeg run as a clean CLIError: the reason is ffmpeg's last
    stderr line (earlier noise dropped), or the exit code when it said nothing."""
    detail = result.stderr.strip().splitlines()
    reason = detail[-1] if detail else f"ffmpeg exited with code {result.returncode}"
    return CLIError(
        f"Could not {action} {dest.name}: {reason}",
        error_type=error_type,
        suggestion="Check that the input is a readable audio/video file.",
    )


def resolve_diarized_transcript(
    api_key: str,
    transcript_id: str | None,
    media: Path,
    *,
    status_message: str,
    json_mode: bool,
    quiet: bool,
) -> object:
    """The diarized transcript driving the command: fetched by id, or made fresh
    from the (already local) media file — always with speaker labels, so the
    caller can select or voice content per speaker."""
    if transcript_id is not None:
        return client.get_transcript(api_key, transcript_id)
    config = aai.TranscriptionConfig(speaker_labels=True)
    with output.status(status_message, json_mode=json_mode, quiet=quiet):
        return client.transcribe(api_key, str(media), config=config)
