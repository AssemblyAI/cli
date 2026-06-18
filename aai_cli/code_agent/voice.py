"""Voice I/O for `assembly code`: speak your request, hear the reply.

The coding agent's default interactive mode (a TTY) captures one spoken turn via
streaming STT and reads each assistant reply back via streaming TTS. Both legs are
injected so the loop is unit-tested with fakes — no microphone, speaker, or socket.

Readback needs streaming TTS, which only the sandbox environment exposes
(`tts.session.is_available`); in production, voice *input* still works and replies
stay on screen as text. Microphone (STT) input works in every environment.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aai_cli.core import client, config_builder
from aai_cli.core.microphone import MicrophoneSource
from aai_cli.tts import session as tts_session
from aai_cli.tts.audio import PcmPlayer
from aai_cli.tts.session import SpeakConfig

if TYPE_CHECKING:
    from assemblyai.streaming.v3 import StreamingParameters

# The audio-device CLIError types listen() raises when no usable microphone is present;
# the command degrades to typed input on these (see _exec._voice_read_line). They mirror
# the error_type values core.microphone attaches to its mic-open failures.
AUDIO_ERROR_TYPES = frozenset({"mic_missing", "mic_error", "audio_input_error"})

# Streaming TTS synthesizes at 24 kHz, the rate the readback player is opened at.
_TTS_SAMPLE_RATE = 24000

# The streaming STT model used to transcribe a spoken turn — the same realtime default
# `assembly stream` and `assembly agent-cascade` use.
_SPEECH_MODEL = "u3-rt-pro"


class Microphone(Protocol):
    """The microphone slice the listen loop drives: an iterable of PCM at a known rate."""

    sample_rate: int

    def __iter__(self) -> Iterator[bytes]:
        """Yield captured PCM16 chunks until the stream ends."""


class StreamFn(Protocol):
    """The streaming-STT call: ``client.stream_audio`` satisfies it structurally."""

    def __call__(
        self,
        api_key: str,
        source: Iterable[bytes],
        *,
        params: StreamingParameters,
        on_turn: Callable[[object], None],
    ) -> None:
        """Stream ``source`` and forward each Turn event to ``on_turn``."""


class SynthFn(Protocol):
    """The streaming-TTS call: ``tts.session.synthesize`` satisfies it structurally.

    The return is typed ``object`` because the readback path discards it (it plays each
    chunk through ``on_audio`` as it arrives), which also lets a test inject a fake that
    returns nothing meaningful.
    """

    def __call__(
        self,
        api_key: str,
        config: SpeakConfig,
        *,
        on_audio: Callable[[bytes, int], None],
    ) -> object:
        """Synthesize ``config.text``, handing each PCM chunk to ``on_audio``."""


class Player(Protocol):
    """The readback player: a context manager that ``feed``s PCM chunks (PcmPlayer)."""

    def __enter__(self) -> Player:
        """Enter the playback context (opens the device lazily on first feed)."""

    def __exit__(self, exc_type: object, *exc: object) -> object:
        """Drain on a clean exit, abort otherwise; never suppress."""

    def feed(self, pcm: bytes, sample_rate: int) -> None:
        """Play one PCM chunk, opening the output device on the first call."""


def _stt_params(sample_rate: int) -> StreamingParameters:
    """StreamingParameters for capturing one spoken turn at ``sample_rate``.

    ``format_turns`` is on so the finalized turn reads like a typed prompt (punctuated
    and cased) rather than raw lowercase tokens.
    """
    merged = config_builder.merge_streaming_params(
        flags={"speech_model": _SPEECH_MODEL, "format_turns": True, "sample_rate": sample_rate}
    )
    return config_builder.construct_streaming_params(merged)


@dataclass
class VoiceSession:
    """Speak-to-it / read-it-back I/O for one coding session, with injectable legs."""

    api_key: str
    readback: bool
    mic_factory: Callable[[], Microphone] = MicrophoneSource
    stream_fn: StreamFn = client.stream_audio
    synth_fn: SynthFn = tts_session.synthesize
    player_factory: Callable[[], Player] = PcmPlayer

    def listen(self) -> str | None:
        """Capture one spoken turn and return its finalized transcript.

        Returns the text of the first end-of-turn the server finalizes, or ``None`` when
        the microphone stream ends without one (EOF — e.g. a finite source in tests). The
        microphone is gated shut the moment a turn finalizes, so exactly one utterance is
        captured per call; a real mic blocks until you speak (Ctrl-C to quit).
        """
        mic = self.mic_factory()
        done = threading.Event()
        captured: list[str] = []

        def on_turn(event: object) -> None:
            text = (getattr(event, "transcript", "") or "").strip()
            if text and getattr(event, "end_of_turn", False):
                captured.append(text)
                done.set()

        def gated() -> Iterator[bytes]:
            for chunk in mic:
                if done.is_set():
                    return
                yield chunk

        self.stream_fn(self.api_key, gated(), params=_stt_params(mic.sample_rate), on_turn=on_turn)
        return " ".join(captured).strip() or None

    def speak(self, text: str) -> None:
        """Read ``text`` back via streaming TTS, when readback is available.

        A no-op when readback is off (production, where streaming TTS has no host) or the
        text is blank — so the caller can route every assistant reply here unconditionally.
        """
        text = text.strip()
        if not self.readback or not text:
            return
        config = SpeakConfig(text=text, sample_rate=_TTS_SAMPLE_RATE)
        with self.player_factory() as player:
            self.synth_fn(self.api_key, config, on_audio=player.feed)


def build_voice_session(api_key: str) -> VoiceSession:
    """A voice session for the active environment.

    Readback is enabled only where streaming TTS is available (the sandbox); microphone
    input is wired regardless.
    """
    return VoiceSession(api_key=api_key, readback=tts_session.is_available())
