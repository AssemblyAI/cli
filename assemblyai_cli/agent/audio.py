from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from assemblyai_cli.errors import CLIError
from assemblyai_cli.microphone import _FALLBACK_RATE, _resample, audio_missing_error

SAMPLE_RATE = 24000  # Voice Agent native PCM16 mono rate


def _output_default_rate(device: int | None = None) -> int:
    """The output device's native sample rate.

    Like the mic, the speaker is opened at its own rate to avoid CoreAudio
    'paramErr' (-50) from forcing an unsupported one; agent audio (24 kHz) is
    resampled to it. Falls back to a safe default when the device can't be queried.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    try:
        rate = int(sd.query_devices(device, "output")["default_samplerate"])
    except Exception:  # noqa: BLE001 - any query failure -> safe fallback, never crash here
        return _FALLBACK_RATE
    return rate if rate > 0 else _FALLBACK_RATE


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
        output_rate: int | None = None,
        rate_query: Callable[[int | None], int] | None = None,
    ) -> None:
        self._source_rate = sample_rate  # rate of enqueued audio (agent = 24 kHz)
        self._factory = stream_factory or _default_output_stream
        query = rate_query or _output_default_rate
        # Open the speaker at its native rate; resample agent audio to it.
        self._device_rate = output_rate if output_rate is not None else query(None)
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        # sounddevice stream (or a test double); typed Any since sounddevice ships no stubs.
        self._stream: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stream = self._factory(self._device_rate)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        state: Any = None
        while True:
            chunk = self._queue.get()
            if chunk is None:
                return
            if self._device_rate != self._source_rate:
                chunk, state = _resample(
                    chunk, state, src_rate=self._source_rate, dst_rate=self._device_rate
                )
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


def _default_duplex_stream(*, rate: int, blocksize: int, callback: Any, device: int | None) -> Any:
    """Open ONE started full-duplex sounddevice stream (mic + speaker together)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    try:
        stream = sd.RawStream(
            samplerate=rate,
            device=device,
            channels=1,
            dtype="int16",
            blocksize=blocksize,
            callback=callback,
        )
        stream.start()
    except Exception as exc:
        raise CLIError(
            f"Could not open the audio device: {exc}",
            error_type="audio_output_error",
            exit_code=1,
        ) from exc
    return stream


class DuplexAudio:
    """Capture and playback over a single full-duplex stream.

    macOS AUHAL refuses two separate input+output streams on one device
    ("cannot do in current context"), which silently kills capture. Driving both
    directions through one `sd.RawStream` callback avoids that. Audio is captured
    at the device's native rate and resampled to `target_rate` (the agent's 24 kHz)
    for the mic side; playback is resampled back to the device rate. Exposes a
    `Player`-compatible `player` and an iterable `mic` so `run_session` is unchanged.
    """

    def __init__(
        self,
        *,
        target_rate: int = SAMPLE_RATE,
        device: int | None = None,
        device_rate: int | None = None,
        stream_factory: Callable[..., Any] | None = None,
        rate_query: Callable[[int | None], int] | None = None,
    ) -> None:
        query = rate_query or _output_default_rate
        self._device_rate = device_rate if device_rate is not None else query(device)
        self._target = target_rate
        self._device = device
        self._factory = stream_factory or _default_duplex_stream
        self._blocksize = max(1, self._device_rate // 10)  # ~100 ms
        self._in: queue.Queue[bytes | None] = queue.Queue()
        self._out = bytearray()  # device-rate playback bytes
        self._out_state: Any = None  # ratecv state for target -> device
        self._lock = threading.Lock()
        self._stream: Any = None
        self._started = False
        self.player = _DuplexPlayer(self)
        self.mic = _DuplexMic(self)

    def _callback(self, indata: Any, outdata: Any, _frames: int, _time: Any, _status: Any) -> None:
        # Capture: hand the device-rate input bytes to the mic consumer.
        with contextlib.suppress(Exception):
            self._in.put_nowait(bytes(indata))
        # Playback: drain the buffer into the output, zero-filling any shortfall.
        need = len(outdata)
        with self._lock:
            take = bytes(self._out[:need])
            del self._out[:need]
        if len(take) == need:
            outdata[:] = take
        else:
            outdata[: len(take)] = take
            outdata[len(take) :] = b"\x00" * (need - len(take))

    def start(self) -> None:
        if self._started:
            return
        self._stream = self._factory(
            rate=self._device_rate,
            blocksize=self._blocksize,
            callback=self._callback,
            device=self._device,
        )
        self._started = True

    def feed(self, pcm: bytes) -> None:
        """Queue target-rate PCM for playback, resampled to the device rate."""
        if self._device_rate != self._target:
            pcm, self._out_state = _resample(
                pcm, self._out_state, src_rate=self._target, dst_rate=self._device_rate
            )
        with self._lock:
            self._out += pcm

    def flush(self) -> None:
        with self._lock:
            self._out.clear()

    def pending(self) -> int:
        with self._lock:
            return len(self._out) // 2

    def capture_frames(self) -> Iterator[bytes]:
        """Yield target-rate PCM captured from the device until closed."""
        state: Any = None
        while True:
            chunk = self._in.get()
            if chunk is None:
                return
            if self._device_rate != self._target:
                chunk, state = _resample(
                    chunk, state, src_rate=self._device_rate, dst_rate=self._target
                )
            yield chunk

    def close(self) -> None:
        self._in.put(None)  # end capture_frames()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
            with contextlib.suppress(Exception):
                self._stream.close()
        self._started = False


class _DuplexPlayer:
    """Player-compatible facade over a DuplexAudio's playback side."""

    def __init__(self, duplex: DuplexAudio) -> None:
        self._duplex = duplex

    def start(self) -> None:
        self._duplex.start()

    def enqueue(self, pcm: bytes) -> None:
        self._duplex.feed(pcm)

    def flush(self) -> None:
        self._duplex.flush()

    def pending(self) -> int:
        return self._duplex.pending()

    def close(self) -> None:
        self._duplex.close()


class _DuplexMic:
    """Iterable of captured target-rate PCM from a DuplexAudio."""

    def __init__(self, duplex: DuplexAudio) -> None:
        self._duplex = duplex

    def __iter__(self) -> Iterator[bytes]:
        return self._duplex.capture_frames()


# Microphone capture (MicrophoneSource) lives in assemblyai_cli.microphone and is
# shared with `aai stream`; the agent's live mic+speaker run through DuplexAudio.
