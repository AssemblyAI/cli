from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import time
import wave
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import Any

from aai_cli.errors import APIError, CLIError

TARGET_RATE = 16000
PCM16_SAMPLE_WIDTH_BYTES = 2
CHUNK_BYTES = TARGET_RATE * PCM16_SAMPLE_WIDTH_BYTES // 10  # 100 ms of 16-bit mono PCM


def _is_streamable_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getnchannels() == 1
                and w.getsampwidth() == PCM16_SAMPLE_WIDTH_BYTES
                and w.getframerate() == TARGET_RATE
            )
    except (wave.Error, EOFError, OSError):
        return False


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


class FileSource:
    """Yields real-time-paced 16 kHz mono PCM chunks from a local file or a URL."""

    def __init__(self, source: str, *, sleep: Callable[[float], object] = time.sleep) -> None:
        self.source = source
        self._sleep = sleep
        self.sample_rate = TARGET_RATE
        # Local paths get a fast WAV path and an existence check; URLs always decode
        # through ffmpeg (which reads http/https inputs natively).
        self._path = None if _is_url(source) else Path(source)
        if self._path is not None:
            if not self._path.is_file():
                raise CLIError(
                    f"No such file: {self._path}",
                    error_type="file_not_found",
                    exit_code=2,
                    suggestion="Check the path, or pass a URL or YouTube link instead.",
                )
            self._wav = _is_streamable_wav(self._path)
        else:
            self._wav = False
        if not self._wav and shutil.which("ffmpeg") is None:
            raise CLIError(
                "This audio source needs ffmpeg.",
                error_type="ffmpeg_missing",
                exit_code=2,
                suggestion="Install ffmpeg, or pass a 16 kHz mono 16-bit WAV.",
            )

    def __iter__(self) -> Iterator[bytes]:
        chunks = self._wav_chunks() if self._wav else self._ffmpeg_chunks()
        produced = 0
        # Closing this outer generator early raises GeneratorExit at the `yield` below,
        # but a plain `for` loop won't forward it into `chunks`; its cleanup (ffmpeg
        # terminate/wait) would then run only at GC, not synchronously. Close it here so
        # the subprocess teardown happens deterministically on early stop.
        try:
            for chunk in chunks:
                produced += len(chunk)  # pragma: no mutate (only == 0 matters)
                yield chunk
                # ~real-time pacing; tests inject a no-op sleep, so the duration is
                # deliberately not asserted (it's cosmetic, not behavioral).
                self._sleep(len(chunk) / (TARGET_RATE * 2))  # pragma: no mutate
        finally:
            chunks.close()
        if produced == 0:
            raise CLIError(
                f"No audio data in {self.source}.",
                error_type="empty_audio",
                exit_code=2,
                suggestion="Check the file isn't empty or silent.",
            )

    def _wav_chunks(self) -> Generator[bytes, None, None]:
        frames_per_chunk = CHUNK_BYTES // 2
        with wave.open(str(self._path), "rb") as w:  # _wav implies a local path
            while True:
                data = w.readframes(frames_per_chunk)
                if not data:
                    return
                yield data

    def _ffmpeg_chunks(self) -> Generator[bytes, None, None]:
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                self.source,  # a local path or an http(s) URL; ffmpeg reads both
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
        completed = False
        try:
            while True:
                data = stdout.read(CHUNK_BYTES)
                if not data:
                    break
                yield data
            completed = True  # natural EOF: let ffmpeg exit on its own
        finally:
            # SIGTERM only on early stop (generator close) or error — terminating a
            # process that already finished would surface as a spurious exit -15.
            if not completed:
                proc.terminate()
            with contextlib.suppress(Exception):
                stdout.close()
            try:
                proc.wait()
            except KeyboardInterrupt:
                # The generator can be finalized late, during an interrupted
                # shutdown; a stray Ctrl-C in wait() must not surface as the noisy
                # "Exception ignored in generator". Kill the child and stay quiet.
                with contextlib.suppress(Exception):
                    proc.kill()
        # Reached only on natural EOF (not early generator close): surface a real
        # decode failure instead of silently streaming nothing.
        if proc.returncode:
            detail = proc.stderr.read().decode("utf-8", "replace").strip() if proc.stderr else ""
            raise APIError(
                f"ffmpeg could not decode {self.source}: {detail or f'exit {proc.returncode}'}"
            )


class StdinSource:
    """Streams raw PCM16 mono audio piped on stdin.

    Expects signed 16-bit little-endian mono PCM at ``sample_rate`` (default 16 kHz):
    ``ffmpeg -i in.mp4 -f s16le -acodec pcm_s16le -ac 1 -ar 16000 - | aai stream -``.
    """

    def __init__(self, *, sample_rate: int = TARGET_RATE, stdin: Any = None) -> None:
        self.source = "-"
        self.sample_rate = sample_rate
        self._stdin = stdin  # injectable for tests; defaults to sys.stdin.buffer

    def __iter__(self) -> Iterator[bytes]:
        stream: Any = self._stdin
        if stream is None:
            stream = getattr(sys.stdin, "buffer", sys.stdin)
        while True:
            data = stream.read(CHUNK_BYTES)
            if not data:
                return
            yield bytes(data)


# MicrophoneSource (mic capture) lives in aai_cli.microphone and is shared
# with the voice agent; FileSource above is the only streaming-specific source.
