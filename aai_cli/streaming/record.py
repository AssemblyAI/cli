"""Tee streamed PCM to a WAV file — backs `assembly stream --save-audio PATH`.

The whole point is a verbatim recording of exactly the bytes sent to the streaming
API, so a caller (e.g. an ensemble that compares the live turns against an async
re-transcribe) can keep the audio without owning capture itself. The tee never alters
what's transcribed: it writes each chunk to disk and yields it onward unchanged.
"""

from __future__ import annotations

import wave
from collections.abc import Generator, Iterable
from pathlib import Path

from aai_cli.core.errors import CLIError
from aai_cli.streaming.sources import PCM16_SAMPLE_WIDTH_BYTES


def validate_target(path: Path) -> None:
    """Reject a ``--save-audio`` path whose parent directory is missing, before streaming.

    Run before credentials/audio are opened so a bad path reads as a path error up
    front, not after a session has already started recording into the void.
    """
    parent = path.parent
    if not parent.is_dir():
        raise CLIError(
            f"Cannot save audio to {path}: {parent} is not a directory.",
            error_type="save_audio_path",
            exit_code=2,
            suggestion="Create the directory first, or pass a path under an existing one.",
        )


def tee_wav(audio: Iterable[bytes], path: Path, *, rate: int) -> Generator[bytes, None, None]:
    """Yield every PCM16 chunk from ``audio`` unchanged while writing it to ``path`` as WAV.

    The recording is mono 16-bit PCM at ``rate`` — the same shape the streaming API
    receives. The header's length fields are patched when the iterable is exhausted or
    closed early (Ctrl-C raises ``GeneratorExit`` at the ``yield``), so even an
    interrupted run leaves a valid, playable WAV of the audio captured so far.
    """
    try:
        # Open the handle ourselves (rather than letting wave.open(str) do it): a bad
        # path then fails here cleanly, with no half-built Wave_write whose __del__ would
        # later raise an "ignored in __del__" warning during GC.
        handle = path.open("wb")
    except OSError as exc:
        raise CLIError(
            f"Cannot open {path} for writing: {exc}",
            error_type="save_audio_path",
            exit_code=2,
        ) from exc
    try:
        # The Wave_write context manager closes (flushes + patches the length fields from
        # what was actually written) on exit, so the file is a valid WAV even when the
        # generator is closed mid-stream (Ctrl-C). The outer finally then closes the
        # handle we opened — after the patch — since wave only closes handles it opened.
        with wave.open(handle, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(PCM16_SAMPLE_WIDTH_BYTES)
            wav.setframerate(rate)
            for chunk in audio:
                wav.writeframesraw(chunk)
                yield chunk
    finally:
        handle.close()
