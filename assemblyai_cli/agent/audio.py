from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

from assemblyai_cli.errors import CLIError
from assemblyai_cli.microphone import audio_missing_error

SAMPLE_RATE = 24000  # Voice Agent native PCM16 mono rate


def _default_output_stream(rate: int) -> Any:
    """Open a sounddevice PCM16 mono output stream (imported lazily to keep startup fast)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    try:
        stream = sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16")
        stream.start()
    except Exception as exc:
        raise CLIError(
            f"Could not open the audio output device: {exc}",
            error_type="audio_output_error",
            exit_code=1,
        ) from exc
    return stream


class Player:
    """Plays queued PCM16 audio chunks through a speaker output stream."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        stream_factory: Callable[[int], object] | None = None,
    ) -> None:
        self._rate = sample_rate
        self._factory = stream_factory or _default_output_stream
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        # sounddevice stream (or a test double); typed Any since sounddevice ships no stubs.
        self._stream: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stream = self._factory(self._rate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            try:
                self._stream.write(chunk)
            except Exception:  # noqa: BLE001 - stream may be torn down mid-write
                return

    def enqueue(self, pcm: bytes) -> None:
        self._queue.put(pcm)

    def flush(self) -> None:
        """Discard pending audio (barge-in / interruption)."""
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def pending(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._queue.put(None)
        # Stop the stream first so any in-flight write() raises and the worker
        # thread returns promptly, avoiding a teardown race with the join below.
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.close()


class NullPlayer:
    """A Player look-alike that discards audio instead of opening a speaker.

    Used by file-driven agent runs (`aai agent <file>`), which only need the
    transcript events: there is no human listening, and headless/CI hosts have
    no output device for `sounddevice` to open.
    """

    def start(self) -> None:
        pass

    def enqueue(self, pcm: bytes) -> None:
        pass

    def flush(self) -> None:
        pass

    def pending(self) -> int:
        return 0

    def close(self) -> None:
        pass


# Microphone capture (MicrophoneSource) lives in assemblyai_cli.microphone and is
# shared with `aai stream`; this module owns only the speaker-side Player.
