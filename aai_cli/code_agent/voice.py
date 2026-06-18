"""Voice I/O for `assembly code`: speak your request, hear the reply.

The coding agent's default interactive mode (a TTY) captures one spoken turn via
streaming STT and reads each assistant reply back via streaming TTS. Both legs are
injected so the loop is unit-tested with fakes — no microphone, speaker, or socket.

Readback needs streaming TTS, which only the sandbox environment exposes
(`tts.session.is_available`); in production, voice *input* still works and replies
stay on screen as text. Microphone (STT) input works in every environment.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NoReturn, Protocol

from aai_cli.core import client, config_builder, errors
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

# Reading code aloud over TTS is useless, so the readback speaks only the prose. These
# strip fenced and inline code, and the spoken summary is capped so a long reply stays brief.
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]+`")
_MAX_SPOKEN_CHARS = 600  # pragma: no mutate — a cosmetic cap on how much prose is read aloud
_ALL_CODE_READBACK = "I've updated the code — see the transcript for the details."


class _ReadbackInterrupted(errors.CLIError):
    """Internal sentinel: raised inside the readback feed when ``cancel()`` fires mid-playback.

    Subclasses ``CLIError`` so streaming TTS re-raises it unchanged (``synthesize`` passes
    ``CLIError`` straight through), letting ``speak`` abort the player and stop promptly instead
    of draining the rest of the clip. It never reaches the user — ``speak`` always catches it.
    """

    def __init__(self) -> None:
        # No exit_code: speak() always catches this, so the inherited default never surfaces.
        super().__init__("readback interrupted", error_type="readback_interrupted")


def _abort_readback() -> NoReturn:
    """Raise the readback sentinel — the cancel signal ``speak``'s feed acts on mid-playback."""
    raise _ReadbackInterrupted


def spoken_summary(text: str) -> str:
    """Reduce an assistant reply to the prose worth reading aloud.

    Drops fenced and inline code, collapses whitespace, and caps the length. When the reply
    was essentially all code (nothing but blocks), returns a short generic note so the
    readback still says *something* rather than going silent.
    """
    prose = _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", text))
    prose = " ".join(prose.split()).strip()
    if not prose:
        return _ALL_CODE_READBACK
    if len(prose) > _MAX_SPOKEN_CHARS:
        return prose[:_MAX_SPOKEN_CHARS].rstrip() + "…"
    return prose


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
    _cancel: threading.Event = field(
        default_factory=threading.Event,
        init=False,  # pragma: no mutate
    )

    def cancel(self) -> None:
        """Stop an in-flight ``listen``/``speak`` so the current voice activity ends promptly.

        Set from another thread (the TUI's Ctrl-C / Escape, since the legs block on a daemon
        thread): the mic gate in :meth:`listen` and the readback feed in :meth:`speak` both
        check it between chunks, so listening or playback stops within a chunk rather than
        running to completion. Each leg clears it on entry, so a stale cancel never preempts
        the next turn.
        """
        self._cancel.set()

    def listen(self) -> str | None:
        """Capture one spoken turn and return its finalized transcript.

        Returns the text of the first end-of-turn the server finalizes, or ``None`` when
        the microphone stream ends without one (EOF — e.g. a finite source in tests, or a
        :meth:`cancel` mid-capture). The microphone is gated shut the moment a turn finalizes,
        so exactly one utterance is captured per call; a real mic blocks until you speak.
        """
        self._cancel.clear()
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
                if done.is_set() or self._cancel.is_set():
                    return
                yield chunk

        self.stream_fn(self.api_key, gated(), params=_stt_params(mic.sample_rate), on_turn=on_turn)
        return " ".join(captured).strip() or None

    def speak(self, text: str) -> None:
        """Read ``text`` back via streaming TTS, when readback is available.

        A no-op when readback is off (production, where streaming TTS has no host) or the
        text is blank — so the caller can route every assistant reply here unconditionally.
        A :meth:`cancel` from another thread stops playback promptly: the feed raises an
        internal sentinel that aborts the player (discarding buffered audio) and ends synthesis.
        """
        text = text.strip()
        if not self.readback or not text:
            return
        self._cancel.clear()
        config = SpeakConfig(text=text, sample_rate=_TTS_SAMPLE_RATE)
        try:
            with self.player_factory() as player:

                def feed(pcm: bytes, sample_rate: int) -> None:
                    if self._cancel.is_set():
                        _abort_readback()
                    player.feed(pcm, sample_rate)

                self.synth_fn(self.api_key, config, on_audio=feed)
        except _ReadbackInterrupted:
            pass  # cancel() asked us to stop; the player aborted on the way out


def build_voice_session(api_key: str) -> VoiceSession:
    """A voice session for the active environment.

    Readback is enabled only where streaming TTS is available (the sandbox); microphone
    input is wired regardless.
    """
    return VoiceSession(api_key=api_key, readback=tts_session.is_available())
