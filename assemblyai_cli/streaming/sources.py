from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

from assemblyai_cli.errors import APIError, CLIError

TARGET_RATE = 16000
CHUNK_BYTES = TARGET_RATE * 2 // 10  # 100 ms of 16-bit mono PCM


def _is_streamable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == TARGET_RATE
            )
    except (wave.Error, EOFError, OSError):
        return False


class FileSource:
    """Yields real-time-paced 16 kHz mono PCM chunks from an audio file."""

    def __init__(self, path: str, *, sleep: Callable[[float], object] = time.sleep) -> None:
        self.path = Path(path)
        self._sleep = sleep
        self.sample_rate = TARGET_RATE
        if not self.path.is_file():
            raise CLIError(f"No such file: {self.path}", error_type="file_not_found", exit_code=2)
        self._wav = _is_streamable_wav(self.path)
        if not self._wav and shutil.which("ffmpeg") is None:
            raise CLIError(
                "This audio format needs ffmpeg. Install ffmpeg, or pass a 16 kHz mono 16-bit WAV.",
                error_type="ffmpeg_missing",
                exit_code=2,
            )

    def __iter__(self) -> Iterator[bytes]:
        chunks = self._wav_chunks() if self._wav else self._ffmpeg_chunks()
        produced = 0
        for chunk in chunks:
            produced += len(chunk)
            yield chunk
            self._sleep(len(chunk) / (TARGET_RATE * 2))  # ~real-time pacing
        if produced == 0:
            raise CLIError(f"No audio data in {self.path}.", error_type="empty_audio", exit_code=2)

    def _wav_chunks(self) -> Iterator[bytes]:
        frames_per_chunk = CHUNK_BYTES // 2
        with wave.open(str(self.path), "rb") as w:
            while True:
                data = w.readframes(frames_per_chunk)
                if not data:
                    return
                yield data

    def _ffmpeg_chunks(self) -> Iterator[bytes]:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(self.path),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                str(TARGET_RATE),
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # stdout=PIPE guarantees a pipe; bind a local so the type checker narrows it.
        stdout = proc.stdout
        if stdout is None:  # pragma: no cover - defensive; PIPE always yields a stream
            raise APIError("ffmpeg did not expose an output stream.")
        try:
            while True:
                data = stdout.read(CHUNK_BYTES)
                if not data:
                    break
                yield data
        finally:
            proc.terminate()
            with contextlib.suppress(Exception):
                stdout.close()
            proc.wait()
        # Reached only on natural EOF (not early generator close): surface a
        # decode failure instead of silently streaming nothing.
        if proc.returncode:
            detail = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
            raise APIError(
                f"ffmpeg could not decode {self.path}: {detail or f'exit {proc.returncode}'}"
            )


# MicrophoneSource (mic capture) lives in assemblyai_cli.microphone and is shared
# with the voice agent; FileSource above is the only streaming-specific source.
