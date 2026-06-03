from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

from assemblyai_cli.errors import CLIError

_MIC_MISSING_MSG = (
    "Microphone support (PyAudio) is unavailable. Try: pip install --force-reinstall pyaudio"
)


def _default_mic_stream(*, sample_rate: int, device: int | None) -> Iterator[bytes]:
    """The SDK's PyAudio-backed mic stream (imported lazily to keep startup fast)."""
    from assemblyai.extras import MicrophoneStream

    return cast(Iterator[bytes], MicrophoneStream(sample_rate=sample_rate, device_index=device))


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
            raise CLIError(_MIC_MISSING_MSG, error_type="mic_missing", exit_code=2) from exc
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
