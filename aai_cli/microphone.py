from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

from aai_cli.errors import CLIError

with warnings.catch_warnings():
    # audioop is deprecated stdlib on 3.11/3.12 (warning suppressed here) and is
    # provided by the `audioop-lts` package on 3.13+, where it left the stdlib.
    # Imported once at module load so the per-chunk resample path stays hot.
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

# Used when the device's native rate can't be determined (e.g. headless CI).
_FALLBACK_RATE = 48000


def audio_missing_error() -> CLIError:
    """The shared 'sounddevice can't be imported' error for mic and speaker paths."""
    return CLIError(
        "Audio support (sounddevice) is unavailable. Try: pip install --force-reinstall sounddevice",
        error_type="mic_missing",
        exit_code=2,
    )


def _default_rate(kind: str, device: int | None = None) -> int:
    """A device's native sample rate for `kind` ("input" or "output").

    Opening a device at its own rate avoids CoreAudio 'paramErr' (-50) failures
    that happen when it's forced to an unsupported rate. Falls back to a safe
    default if the device can't be queried (no device, headless CI).
    """
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    try:
        rate = int(sd.query_devices(device, kind)["default_samplerate"])
    except Exception:  # noqa: BLE001 - any query failure -> safe fallback, never crash here
        return _FALLBACK_RATE
    return rate if rate > 0 else _FALLBACK_RATE


def _device_default_rate(device: int | None = None) -> int:
    """The input device's native sample rate (see `_default_rate`)."""
    return _default_rate("input", device)


def _resample(chunk: bytes, state: Any, *, src_rate: int, dst_rate: int) -> tuple[bytes, Any]:
    """Resample one PCM16 mono fragment from `src_rate` to `dst_rate`."""
    return audioop.ratecv(chunk, 2, 1, src_rate, dst_rate, state)


class _SoundDeviceMic:
    """Iterator of PCM16 byte chunks from a sounddevice raw input stream.

    Yields ~100 ms blocks; closeable so MicrophoneSource can tear it down.
    """

    def __init__(self, stream: Any, blocksize: int) -> None:
        self._stream = stream
        self._blocksize = blocksize

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        data, _overflowed = self._stream.read(self._blocksize)
        return bytes(data)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def _default_mic_stream(*, sample_rate: int, device: int | None) -> Iterator[bytes]:
    """A sounddevice-backed PCM16 mic stream (imported lazily to keep startup fast)."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc

    blocksize = max(1, sample_rate // 10)  # ~100 ms per read
    stream = sd.RawInputStream(
        samplerate=sample_rate, device=device, channels=1, dtype="int16", blocksize=blocksize
    )
    stream.start()
    return _SoundDeviceMic(stream, blocksize)


class MicrophoneSource:
    """Iterable of PCM16 chunks captured at the microphone's native rate.

    Shared by `aai stream` (mic input) and `aai agent` (captured speech). The
    device is opened at its own sample rate to avoid forcing an unsupported one;
    with `target_rate` set (the voice agent needs 24 kHz) the captured audio is
    resampled to it, otherwise frames are yielded at the capture rate, which
    `sample_rate` reports for the streaming API. The stream factory and rate
    lookup are injectable so tests don't need real audio hardware.
    """

    def __init__(
        self,
        *,
        target_rate: int | None = None,
        device: int | None = None,
        capture_rate: int | None = None,
        stream_factory: Callable[..., Iterator[bytes]] | None = None,
        rate_query: Callable[[int | None], int] | None = None,
        on_open: Callable[[], None] | None = None,
    ) -> None:
        self.device = device
        self.target_rate = target_rate
        # Fired once the device is open and capturing, so callers only announce
        # "listening" when the mic is truly recording — not when the session opens.
        self._on_open = on_open
        self._factory = stream_factory or _default_mic_stream
        query = rate_query or _device_default_rate
        self._capture_rate = capture_rate if capture_rate is not None else query(device)
        # What the yielded PCM is sampled at (resampled to target_rate when set).
        self.sample_rate = target_rate or self._capture_rate

    def __iter__(self) -> Iterator[bytes]:
        try:
            stream: Any = self._factory(sample_rate=self._capture_rate, device=self.device)
        except ImportError as exc:
            raise audio_missing_error() from exc
        except Exception as exc:
            raise CLIError(
                f"Could not open the microphone (device {self.device}): {exc}",
                error_type="mic_error",
                exit_code=1,
            ) from exc
        if self._on_open is not None:
            self._on_open()  # the device is open and recording now
        close = getattr(stream, "close", None)
        state: Any = None
        try:
            for chunk in stream:
                if self.target_rate is not None and self.target_rate != self._capture_rate:
                    chunk, state = _resample(
                        chunk, state, src_rate=self._capture_rate, dst_rate=self.target_rate
                    )
                yield chunk
        finally:
            if callable(close):
                close()
