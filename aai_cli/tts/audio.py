from __future__ import annotations

import contextlib
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from aai_cli.core.errors import CLIError
from aai_cli.core.microphone import import_sounddevice


class _OutputStream(Protocol):
    """The slice of a sounddevice output stream the player drives — named as a
    Protocol so the untyped library boundary is structurally typed, not opaque."""

    def start(self) -> None:
        """Begin playback."""

    def write(self, data: bytes, /) -> object:
        """Queue PCM for playback (the real write returns a bool we ignore)."""

    def stop(self) -> None:
        """Stop after draining buffered frames."""

    def abort(self) -> None:
        """Immediate stop: discards buffered frames (vs stop's drain)."""

    def close(self) -> None:
        """Release the stream."""


# Write playback in ~4 KiB chunks (≈85 ms of 16-bit mono at 24 kHz) instead of one
# big blocking write, so a Ctrl-C is delivered between chunks and cancels promptly
# rather than only after the entire clip has been queued to the output device.
_PLAYBACK_CHUNK_BYTES = 4096


def write_wav(path: Path, pcm: bytes | bytearray, sample_rate: int) -> None:
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
    sd = import_sounddevice()
    stream: _OutputStream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
    return stream


def _playback_error(exc: Exception) -> CLIError:
    return CLIError(
        f"Could not play audio: {exc}",
        error_type="audio_output_error",
        exit_code=1,
        suggestion="Check your output device and run 'assembly doctor', or use --out to save a WAV.",
    )


class PcmPlayer:
    """An incremental 16-bit mono player: ``feed`` each PCM chunk as it is produced.

    Used as a context manager so audio can start on the *first* chunk while later
    chunks are still being synthesized (streaming TTS), instead of waiting for the
    whole clip. The output sample rate isn't known until the server reports it
    mid-stream, so the device is opened lazily on the first ``feed``. On a normal
    exit the stream drains; on Ctrl-C (or any error) it is aborted — buffered
    frames discarded for an immediate stop — and the cancel propagates. Each chunk
    is written in short pieces so a Ctrl-C lands promptly between writes. A device
    failure is wrapped in a clean CLIError that points at --out as the headless
    escape hatch. ``stream_factory`` is injectable for tests.
    """

    def __init__(self, *, stream_factory: Callable[[int], _OutputStream] | None = None) -> None:
        self._factory = stream_factory or _default_output_stream
        self._stream: _OutputStream | None = None

    def __enter__(self) -> PcmPlayer:
        return self

    def feed(self, pcm: bytes, sample_rate: int) -> None:
        """Play one PCM chunk, opening the device on the first chunk."""
        if self._stream is None:
            self._stream = self._open(sample_rate)
        self._write(self._stream, pcm)

    def _open(self, sample_rate: int) -> _OutputStream:
        try:
            stream = self._factory(sample_rate)
            stream.start()
        except CLIError:
            raise  # audio_missing_error() is already user-facing
        except Exception as exc:
            raise _playback_error(exc) from exc
        return stream

    @staticmethod
    def _write(stream: _OutputStream, pcm: bytes) -> None:
        # KeyboardInterrupt (a BaseException) passes through this Exception handler
        # to __exit__, which aborts the device; only real device errors are wrapped.
        try:
            for offset in range(0, len(pcm), _PLAYBACK_CHUNK_BYTES):
                stream.write(pcm[offset : offset + _PLAYBACK_CHUNK_BYTES])
        except Exception as exc:
            raise _playback_error(exc) from exc

    def __exit__(self, exc_type: object, *_: object) -> Literal[False]:  # pragma: no mutate
        stream = self._stream
        if stream is not None:
            try:
                if exc_type is None:  # normal exit -> drain; an error/Ctrl-C -> abort
                    stream.stop()
                else:
                    # Cut sound immediately (discard buffered frames) instead of
                    # letting stop() drain the rest, then let the error propagate.
                    with contextlib.suppress(Exception):
                        stream.abort()
            finally:
                with contextlib.suppress(Exception):
                    stream.close()
        return False  # never suppress: Ctrl-C / device errors must reach the CLI


def play_pcm(
    pcm: bytes,
    sample_rate: int,
    *,
    stream_factory: Callable[[int], _OutputStream] | None = None,
) -> None:
    """Play a complete 16-bit mono PCM buffer through the default output device.

    A thin convenience over ``PcmPlayer`` for callers that already hold the whole
    clip (the multi-voice dialogue path); it blocks until playback finishes.
    """
    with PcmPlayer(stream_factory=stream_factory) as player:
        player.feed(pcm, sample_rate)
