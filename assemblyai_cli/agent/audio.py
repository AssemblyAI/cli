from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator

from assemblyai_cli.errors import CLIError

SAMPLE_RATE = 24000  # Voice Agent native PCM16 mono rate

_MIC_MISSING_MSG = "Audio support isn't installed. Run: pip install 'assemblyai-cli[mic]'"


def _default_output_stream(rate: int):
    """Open a PyAudio PCM16 mono output stream (lazy import; needs the [mic] extra)."""
    try:
        import pyaudio
    except ImportError as exc:
        raise CLIError(_MIC_MISSING_MSG, error_type="mic_missing", exit_code=2) from exc
    try:
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    except Exception as exc:  # noqa: BLE001 - surface device errors cleanly
        raise CLIError(
            f"Could not open the audio output device: {exc}",
            error_type="audio_output_error",
            exit_code=1,
        ) from exc
    stream._pa = pa  # retain so PyAudio isn't GC'd before the stream; terminated in Player.close()
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
        self._stream = None
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
                self._stream.stop_stream()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._stream is not None:
            pa = getattr(self._stream, "_pa", None)
            with contextlib.suppress(Exception):
                self._stream.close()
            if pa is not None:
                with contextlib.suppress(Exception):
                    pa.terminate()


def _default_mic_stream(*, sample_rate: int, device: int | None) -> Iterator[bytes]:
    """SDK PyAudio-backed mic stream (lazy import so the base install stays light)."""
    from assemblyai.extras import MicrophoneStream

    return MicrophoneStream(sample_rate=sample_rate, device_index=device)


class MicCapture:
    """Iterates PCM16 chunks from the microphone (requires the [mic] extra)."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        device: int | None = None,
        stream_factory: Callable[..., Iterator[bytes]] | None = None,
    ) -> None:
        self._rate = sample_rate
        self._device = device
        self._factory = stream_factory or _default_mic_stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            stream = self._factory(sample_rate=self._rate, device=self._device)
        except ImportError as exc:
            raise CLIError(_MIC_MISSING_MSG, error_type="mic_missing", exit_code=2) from exc
        except Exception as exc:  # noqa: BLE001 - surface device errors cleanly
            raise CLIError(
                f"Could not open the microphone (device {self._device}): {exc}",
                error_type="mic_error",
                exit_code=1,
            ) from exc
        close = getattr(stream, "close", None)
        try:
            yield from stream
        finally:
            if callable(close):
                close()
