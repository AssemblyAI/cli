from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from aai_cli.errors import CLIError
from aai_cli.microphone import audio_missing_error


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    """Write 16-bit mono PCM to a WAV file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _default_output_stream(sample_rate: int) -> Any:
    """A started-on-demand raw 16-bit mono output stream from sounddevice."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise audio_missing_error() from exc
    return sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")


def play_pcm(
    pcm: bytes,
    sample_rate: int,
    *,
    stream_factory: Callable[[int], Any] | None = None,
) -> None:
    """Play 16-bit mono PCM through the default output device (blocks until done).

    ``stream_factory`` is injectable for tests; a device failure is wrapped in a
    clean CLIError that points at --out as the headless escape hatch.
    """
    factory = stream_factory or _default_output_stream
    try:
        stream = factory(sample_rate)
        stream.start()
        stream.write(pcm)
        stream.stop()
        stream.close()
    except CLIError:
        raise  # audio_missing_error() is already user-facing
    except Exception as exc:
        raise CLIError(
            f"Could not play audio: {exc}",
            error_type="audio_output_error",
            exit_code=1,
            suggestion="Check your output device and run 'aai doctor', or use --out to save a WAV.",
        ) from exc
