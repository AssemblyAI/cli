from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from assemblyai_cli.errors import CLIError


def audio_missing_error() -> CLIError:
    """The shared 'sounddevice can't be imported' error for mic and speaker paths."""
    return CLIError(
        "Audio support (sounddevice) is unavailable. Try: pip install --force-reinstall sounddevice",
        error_type="mic_missing",
        exit_code=2,
    )


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
    """Iterable of PCM16 chunks from the default microphone.

    Shared by `aai stream` (mic input) and `aai agent` (captured speech). The
    stream factory is injectable so tests don't need real audio hardware.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        device: int | None = None,
        stream_factory: Callable[..., Iterator[bytes]] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._factory = stream_factory or _default_mic_stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            stream: Any = self._factory(sample_rate=self.sample_rate, device=self.device)
        except ImportError as exc:
            raise audio_missing_error() from exc
        except Exception as exc:
            raise CLIError(
                f"Could not open the microphone (device {self.device}): {exc}",
                error_type="mic_error",
                exit_code=1,
            ) from exc
        close = getattr(stream, "close", None)
        try:
            yield from stream
        finally:
            if callable(close):
                close()
