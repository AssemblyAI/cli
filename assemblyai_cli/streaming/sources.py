from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
import wave
from collections.abc import Iterator
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

    def __init__(self, path: str, *, sleep=time.sleep) -> None:
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
        try:
            while True:
                data = proc.stdout.read(CHUNK_BYTES)
                if not data:
                    break
                yield data
        finally:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.stdout.close()
            proc.wait()
        # Reached only on natural EOF (not early generator close): surface a
        # decode failure instead of silently streaming nothing.
        if proc.returncode:
            detail = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
            raise APIError(
                f"ffmpeg could not decode {self.path}: {detail or f'exit {proc.returncode}'}"
            )


def _load_microphone_stream():
    """Import the SDK's PyAudio-backed mic stream (isolated for testing/patching)."""
    from assemblyai.extras import MicrophoneStream

    return MicrophoneStream


class MicSource:
    """Yields PCM chunks from the default microphone (requires the [mic] extra)."""

    def __init__(self, *, sample_rate: int, device: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device

    def __iter__(self) -> Iterator[bytes]:
        try:
            microphone_stream_cls = _load_microphone_stream()
            stream = microphone_stream_cls(sample_rate=self.sample_rate, device_index=self.device)
        except ImportError as exc:
            raise CLIError(
                "Microphone support isn't installed. Run: pip install 'assemblyai-cli[mic]'",
                error_type="mic_missing",
                exit_code=2,
            ) from exc
        except Exception as exc:  # noqa: BLE001 - surface device errors cleanly
            raise CLIError(
                f"Could not open the microphone (device {self.device}): {exc}",
                error_type="mic_error",
                exit_code=1,
            ) from exc
        try:
            yield from stream
        finally:
            stream.close()
