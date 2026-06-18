from __future__ import annotations

import atexit
import contextlib
import signal
import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import ModuleType
from typing import Any, Protocol, cast

from aai_cli.core import stdio
from aai_cli.core.errors import CLIError

with warnings.catch_warnings():
    # audioop is deprecated stdlib on 3.11/3.12 (warning suppressed here) and is
    # provided by the `audioop-lts` package on 3.13+, where it left the stdlib.
    # Imported once at module load so the per-chunk resample path stays hot.
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

# Used when the device's native rate can't be determined (e.g. headless CI).
_FALLBACK_RATE = 48000
# Channel count for the multichannel-input fallback: capture stereo, then downmix to mono.
_STEREO_CHANNELS = 2


class _RawInputStream(Protocol):
    def start(self) -> None:
        """Begin capturing."""

    @abstractmethod
    def read(self, frames: int) -> tuple[bytes, object]:
        """Read up to `frames` frames of PCM plus an overflow flag."""

    def stop(self) -> None:
        """Stop capturing."""

    def close(self) -> None:
        """Release the device."""


class _SoundDeviceModule(Protocol):
    RawInputStream: Callable[..., _RawInputStream]

    @abstractmethod
    def query_devices(
        self, device: int | None = None, kind: str | None = None
    ) -> Mapping[str, object]:
        """Describe an audio device (or the default one for `kind`)."""


def audio_missing_error() -> CLIError:
    """The shared 'sounddevice can't be imported' error for mic and speaker paths."""
    return CLIError(
        "Audio support (sounddevice) is unavailable.",
        error_type="mic_missing",
        exit_code=2,
        suggestion="Reinstall it: pip install --force-reinstall sounddevice",
    )


# Process-global once-latch. The default is only observable on the very first install
# in a fresh process; the suite mutates this flag across tests, so the load-time value
# can't be asserted in isolation — the check/set in _install_… are what the tests pin.
_shutdown_interrupt_guard_installed = False  # pragma: no mutate


def _ignore_interrupt_during_shutdown() -> None:
    """Drop SIGINT for the remainder of interpreter shutdown.

    sounddevice registers its own atexit handler that calls ``Pa_Terminate`` to tear
    down PortAudio. A second Ctrl-C while that runs raises ``KeyboardInterrupt``
    *inside* the atexit callback, which Python reports as a noisy "Exception ignored in
    atexit callback" traceback — even though the first Ctrl-C already stopped the
    session cleanly. There is nothing left to cancel once we're exiting, so ignore the
    late interrupt.
    """
    # signal.signal only works on the main thread; atexit runs there, but a ValueError
    # is still possible in odd embeddings, so guard it rather than crash the teardown.
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGINT, signal.SIG_IGN)


def _install_shutdown_interrupt_guard() -> None:
    """Register ``_ignore_interrupt_during_shutdown`` with atexit exactly once.

    Registered *after* sounddevice imports so atexit's LIFO order runs our guard
    before sounddevice's PortAudio teardown, neutralizing a second Ctrl-C that would
    otherwise raise inside that atexit callback.
    """
    global _shutdown_interrupt_guard_installed
    if _shutdown_interrupt_guard_installed:
        return
    atexit.register(_ignore_interrupt_during_shutdown)
    _shutdown_interrupt_guard_installed = True


def import_sounddevice() -> ModuleType:
    """Import sounddevice lazily, mapping an ImportError to ``audio_missing_error``.

    The one import-and-fail path for every audio device opener (mic capture,
    TTS playback, the agent's duplex stream), so a broken sounddevice install
    yields the same actionable error no matter which command hit it first.
    """
    try:
        import sounddevice
    except ImportError as exc:
        raise audio_missing_error() from exc
    _install_shutdown_interrupt_guard()
    module: ModuleType = sounddevice
    return module


def _sounddevice() -> _SoundDeviceModule:
    return cast("_SoundDeviceModule", import_sounddevice())


def default_rate(kind: str, device: int | None = None) -> int:
    """A device's native sample rate for `kind` ("input" or "output").

    Opening a device at its own rate avoids CoreAudio 'paramErr' (-50) failures
    that happen when it's forced to an unsupported rate. Falls back to a safe
    default if the device can't be queried (no device, headless CI).
    """
    sd = _sounddevice()
    try:
        # query_devices triggers PortAudio's lazy init, which prints device-probe noise to
        # the C-level stderr; suppress it so a TUI mic-open can't corrupt the rendered screen.
        with stdio.suppress_native_stderr():
            devices = sd.query_devices(device, kind)
        raw_rate = devices.get("default_samplerate", _FALLBACK_RATE)
        if not isinstance(raw_rate, str | int | float):
            return _FALLBACK_RATE
        rate = int(float(raw_rate))
    except Exception:  # noqa: BLE001 - any query failure -> safe fallback, never crash here
        return _FALLBACK_RATE
    return rate if rate > 0 else _FALLBACK_RATE


def _device_default_rate(device: int | None = None) -> int:
    """The input device's native sample rate (see `default_rate`)."""
    return default_rate("input", device)


def resample_pcm16(chunk: bytes, state: Any, *, src_rate: int, dst_rate: int) -> tuple[bytes, Any]:
    """Resample one PCM16 mono fragment from `src_rate` to `dst_rate`."""
    return audioop.ratecv(chunk, 2, 1, src_rate, dst_rate, state)


class _SoundDeviceMic:
    """Iterator of PCM16 byte chunks from a sounddevice raw input stream.

    Yields ~100 ms blocks; closeable so MicrophoneSource can tear it down. When opened with
    ``channels=2`` (the multichannel-input fallback below), each interleaved stereo block is
    downmixed to mono so downstream — resampling and the STT stream — always sees one channel.
    """

    def __init__(self, stream: _RawInputStream, blocksize: int, *, channels: int = 1) -> None:
        self._stream = stream
        self._blocksize = blocksize
        self._channels = channels

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        data, _overflowed = self._stream.read(self._blocksize)
        pcm = bytes(data)
        if self._channels == _STEREO_CHANNELS:
            # Average L/R into a single channel (width=2 → int16).
            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
        return pcm

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def _open_input_stream(
    sd: _SoundDeviceModule, *, sample_rate: int, device: int | None, channels: int, blocksize: int
) -> _RawInputStream:
    """Open and start a started PCM16 input stream at ``channels`` channels.

    Wrapped in ``suppress_native_stderr`` because opening/starting is PortAudio's stderr-noisy
    moment — kept off the terminal so a TUI mic-open can't corrupt the rendered screen.
    """
    with stdio.suppress_native_stderr():
        stream = sd.RawInputStream(
            samplerate=sample_rate,
            device=device,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
        )
        stream.start()
    return stream


def _max_input_channels(sd: _SoundDeviceModule, device: int | None) -> int:
    """The device's advertised input-channel count (0 when it exposes no input)."""
    with stdio.suppress_native_stderr():
        info = sd.query_devices(device, "input")
    raw = info.get("max_input_channels", 0)
    return raw if isinstance(raw, int) else 0


def _default_mic_stream(*, sample_rate: int, device: int | None) -> Iterator[bytes]:
    """A sounddevice-backed PCM16 mono mic stream (imported lazily to keep startup fast).

    Tries a mono open first. PortAudio rejects ``channels=1`` (``-9998``) when the device
    exposes no usable mono input: either it has zero input channels (no mic permission, or the
    default input isn't a microphone) — which no channel count can fix, so we raise an
    actionable error — or it's a multichannel-only input, which we reopen at stereo and
    downmix. Devices that already do mono never reach the fallback.
    """
    sd = _sounddevice()
    blocksize = max(1, sample_rate // 10)  # ~100 ms per read
    try:
        return _SoundDeviceMic(
            _open_input_stream(
                sd, sample_rate=sample_rate, device=device, channels=1, blocksize=blocksize
            ),
            blocksize,
        )
    except Exception:
        max_in = _max_input_channels(sd, device)
        if max_in < 1:
            raise CLIError(
                "The default microphone reports no input channels.",
                error_type="mic_error",
                exit_code=1,
                suggestion=(
                    "Grant microphone access to your terminal in System Settings > Privacy & "
                    "Security > Microphone, or pick another input with --device."
                ),
            ) from None
        if max_in < _STEREO_CHANNELS:
            raise  # a 1-channel device should accept mono; surface the real PortAudio error
        stream = _open_input_stream(
            sd,
            sample_rate=sample_rate,
            device=device,
            channels=_STEREO_CHANNELS,
            blocksize=blocksize,
        )
        return _SoundDeviceMic(stream, blocksize, channels=_STEREO_CHANNELS)


class MicrophoneSource:
    """Iterable of PCM16 chunks captured at the microphone's native rate.

    Shared by `assembly stream` (mic input) and `assembly agent` (captured speech). The
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
        stream_factory: Callable[..., Iterable[bytes]] | None = None,
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
        except CLIError:
            raise  # the factory already raised an actionable error; don't bury it in a re-wrap
        except Exception as exc:
            # "device None" reads like a bug; name the default mic in plain words.
            target = (
                "the default microphone"
                if self.device is None
                else f"microphone device {self.device}"
            )
            raise CLIError(
                f"Could not open {target}: {exc}",
                error_type="mic_error",
                exit_code=1,
                suggestion=(
                    "Check your OS microphone permissions for this terminal, or pick "
                    "another input with --device (list devices: python -m sounddevice)."
                ),
            ) from exc
        if self._on_open is not None:
            self._on_open()  # the device is open and recording now
        close = getattr(stream, "close", None)
        state: Any = None
        try:
            for chunk in stream:
                out = chunk
                if self.target_rate is not None and self.target_rate != self._capture_rate:
                    out, state = resample_pcm16(
                        chunk, state, src_rate=self._capture_rate, dst_rate=self.target_rate
                    )
                yield out
        finally:
            if callable(close):
                close()
