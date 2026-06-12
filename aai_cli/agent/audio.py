from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

from aai_cli.errors import CLIError
from aai_cli.microphone import default_rate, import_sounddevice, resample_pcm16

SAMPLE_RATE = 24000  # Voice Agent native PCM16 mono rate


def _output_default_rate(device: int | None = None) -> int:
    """The output device's native sample rate.

    Like the mic, the speaker is opened at its own rate to avoid CoreAudio
    'paramErr' (-50) from forcing an unsupported one; agent audio (24 kHz) is
    resampled to it. Falls back to a safe default when the device can't be queried.
    """
    return default_rate("output", device)


class NullPlayer:
    """A player look-alike that discards audio instead of opening a speaker.

    Used by file-driven agent runs (`assembly agent <file>`), which only need the
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
    sd = import_sounddevice()
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
            suggestion="Check your microphone/output device, then run 'assembly doctor'.",
        ) from exc
    return stream


class DuplexAudio:
    """Capture and playback over a single full-duplex stream.

    macOS AUHAL refuses two separate input+output streams on one device
    ("cannot do in current context"), which silently kills capture. Driving both
    directions through one `sd.RawStream` callback avoids that. Audio is captured
    at the device's native rate and resampled to `target_rate` (the agent's 24 kHz)
    for the mic side; playback is resampled back to the device rate. Exposes a
    player-compatible `player` and an iterable `mic` so `run_session` is unchanged.
    """

    def __init__(
        self,
        *,
        target_rate: int = SAMPLE_RATE,
        device: int | None = None,
        device_rate: int | None = None,
        stream_factory: Callable[..., Any] | None = None,
        rate_query: Callable[[int | None], int] | None = None,
        poll_timeout: float = 1.0,
    ) -> None:
        query = rate_query or _output_default_rate
        self._device_rate = device_rate if device_rate is not None else query(device)
        self._target = target_rate
        self._device = device
        self._factory = stream_factory or _default_duplex_stream
        self._blocksize = max(1, self._device_rate // 10)  # ~100 ms
        # Thread ownership: `_in` is a thread-safe Queue handed from the PortAudio
        # callback thread to capture_frames(). `_out` (device-rate playback bytes) is
        # shared between feed()/flush() (caller thread) and the callback, so every
        # access goes through `_lock`. `_out_state` (the target->device ratecv state)
        # is touched ONLY by feed(), never the callback, so it needs no lock.
        self._in: queue.Queue[bytes | None] = queue.Queue()
        # How long capture_frames() waits for a chunk before checking whether the
        # device stream silently died (e.g. unplugged); injectable for fast tests.
        self._poll_timeout = poll_timeout
        self._out = bytearray()
        self._out_state: Any = None
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
            pcm, self._out_state = resample_pcm16(
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
        """Yield target-rate PCM captured from the device until closed.

        Waits in short timeouts rather than blocking forever: if the PortAudio
        stream dies without close() being called (device unplugged mid-session),
        no sentinel ever arrives, so a bare get() would hang the capture thread —
        and with it the whole agent session — instead of surfacing an error.
        """
        state: Any = None
        while True:
            try:
                chunk = self._in.get(timeout=self._poll_timeout)
            except queue.Empty:
                if self._started and not getattr(self._stream, "active", True):
                    raise CLIError(
                        "The audio device stopped producing input.",
                        error_type="audio_input_error",
                        exit_code=1,
                        suggestion="Check your microphone/output device, then run 'assembly doctor'.",
                    ) from None
                continue
            if chunk is None:
                return
            if self._device_rate != self._target:
                chunk, state = resample_pcm16(
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
    """A player-compatible facade over a DuplexAudio's playback side."""

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


# Microphone capture (MicrophoneSource) lives in aai_cli.microphone and is
# shared with `assembly stream`; the agent's live mic+speaker run through DuplexAudio.
