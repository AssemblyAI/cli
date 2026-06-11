from __future__ import annotations

import contextlib
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aai_cli.errors import CLIError
from aai_cli.microphone import audio_missing_error


class _OutputStream(Protocol):
    """The slice of a sounddevice output stream play_pcm drives — named as a
    Protocol so the untyped library boundary is structurally typed, not opaque."""

    def start(self) -> None: ...
    def write(self, data: bytes, /) -> object: ...  # real write returns a bool we ignore
    def stop(self) -> None: ...
    def abort(self) -> None: ...  # immediate stop: discards buffered frames (vs stop's drain)
    def close(self) -> None: ...


# Write playback in ~4 KiB chunks (≈85 ms of 16-bit mono at 24 kHz) instead of one
# big blocking write, so a Ctrl-C is delivered between chunks and cancels promptly
# rather than only after the entire clip has been queued to the output device.
_PLAYBACK_CHUNK_BYTES = 4096


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write 16-bit mono PCM to a WAV file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def silence(sample_rate: int, seconds: float) -> bytes:
    """Zeroed 16-bit mono PCM of the given duration (2 bytes per frame)."""
    return b"\x00" * (int(sample_rate * seconds) * 2)


def _default_output_stream(sample_rate: int) -> _OutputStream:
    """A started-on-demand raw 16-bit mono output stream from sounddevice."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    stream: _OutputStream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
    return stream


def _playback_error(exc: Exception) -> CLIError:
    return CLIError(
        f"Could not play audio: {exc}",
        error_type="audio_output_error",
        exit_code=1,
        suggestion="Check your output device and run 'aai doctor', or use --out to save a WAV.",
    )


def play_pcm(
    pcm: bytes,
    sample_rate: int,
    *,
    stream_factory: Callable[[int], _OutputStream] | None = None,
) -> None:
    """Play 16-bit mono PCM through the default output device (blocks until done).

    Audio is written in short chunks so a Ctrl-C interrupts promptly: on
    KeyboardInterrupt the stream is aborted (buffered frames discarded) for an
    immediate stop, then the cancel propagates. ``stream_factory`` is injectable
    for tests; a device failure is wrapped in a clean CLIError that points at
    --out as the headless escape hatch.
    """
    factory = stream_factory or _default_output_stream
    try:
        stream = factory(sample_rate)
    except CLIError:
        raise  # audio_missing_error() is already user-facing
    except Exception as exc:
        raise _playback_error(exc) from exc

    try:
        stream.start()
        for offset in range(0, len(pcm), _PLAYBACK_CHUNK_BYTES):
            stream.write(pcm[offset : offset + _PLAYBACK_CHUNK_BYTES])
        stream.stop()
    except KeyboardInterrupt:
        # Cut sound immediately (discard whatever is still buffered in the device)
        # instead of letting stop() drain the rest, then propagate the cancel.
        with contextlib.suppress(Exception):
            stream.abort()
        raise
    except Exception as exc:
        raise _playback_error(exc) from exc
    finally:
        with contextlib.suppress(Exception):
            stream.close()
