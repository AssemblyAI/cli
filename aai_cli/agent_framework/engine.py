"""The terminal voice cascade: Streaming STT -> LLM Gateway -> streaming TTS.

``run_cascade`` greets the user, then drives a live conversation by reading STT
turns and, for each finalized turn, streaming an LLM reply out through TTS
sentence-by-sentence. A new turn barges in on a reply that is still playing.

All three network legs are injected through ``CascadeDeps`` (the same seam
``aai_cli/tts/session.py`` uses), so the orchestration is unit-tested against
fakes with no sockets, microphone, or speaker.
"""

from __future__ import annotations

import contextlib
import threading
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from aai_cli.agent_framework.config import CascadeConfig
from aai_cli.agent_framework.text import split_sentences, trim_history
from aai_cli.core import client, config_builder, llm
from aai_cli.core.errors import CLIError
from aai_cli.tts import session as tts_session
from aai_cli.tts.session import SpeakConfig
from aai_cli.ui import output

if TYPE_CHECKING:
    from assemblyai.streaming.v3 import StreamingParameters
    from openai.types.chat import ChatCompletionMessageParam

# Streaming TTS synthesizes at 24 kHz, the rate the live player is opened at.
TTS_SAMPLE_RATE = 24000


class _Worker(Protocol):
    """The slice of a thread the session drives: started already, queryable, joinable."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Whether the reply worker is still running."""

    def join(self) -> None:
        """Block until the reply worker finishes."""


class Renderer(Protocol):
    """The conversation-rendering surface the cascade drives (AgentRenderer satisfies it)."""

    def connected(self) -> None:
        """Announce the session is live and listening."""

    def user_partial(self, text: str) -> None:
        """Show an interim user transcript."""

    def user_final(self, text: str) -> None:
        """Show a finalized user transcript."""

    def reply_started(self) -> None:
        """Mark the start of an agent reply."""

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        """Show a line of the agent's reply."""

    def reply_done(self, *, interrupted: bool) -> None:
        """Mark the end of an agent reply."""


class Player(Protocol):
    """The speaker the cascade enqueues TTS audio into (DuplexAudio/NullPlayer satisfy it)."""

    def start(self) -> None:
        """Open the output stream."""

    def enqueue(self, pcm: bytes) -> None:
        """Queue PCM audio for playback."""

    def flush(self) -> None:
        """Drop any queued-but-unplayed audio (used on barge-in)."""

    def close(self) -> None:
        """Close the output stream."""


def _new_history() -> list[ChatCompletionMessageParam]:
    """Typed empty-history factory (ChatCompletionMessageParam is import-time-only)."""
    return []


def _spawn_thread(target: Callable[[], None]) -> _Worker:
    """Start ``target`` on a daemon thread so a reply is generated without blocking
    the STT reader (which must stay free to detect a barge-in)."""
    thread = threading.Thread(target=target, daemon=True)  # pragma: no mutate
    thread.start()
    return thread


# The realtime model the cascade transcribes with (same as the agent-framework template).
STT_SPEECH_MODEL = "u3-rt-pro"


def _stt_params(sample_rate: int) -> StreamingParameters:
    """Streaming v3 params for the cascade: PCM at ``sample_rate`` with formatted turns
    (so ``turn_is_formatted`` marks the cue to reply)."""
    merged = config_builder.merge_streaming_params(
        flags={
            "sample_rate": sample_rate,
            "format_turns": True,
            "speech_model": STT_SPEECH_MODEL,
        }
    )
    return config_builder.construct_streaming_params(merged)


@dataclass
class CascadeDeps:
    """The cascade's three network legs plus its thread spawner, all injectable.

    ``CascadeDeps.real`` wires the live STT/LLM/TTS clients; tests pass fakes with
    the same shapes (and a synchronous ``spawn``) to drive the orchestration.
    """

    run_stt: Callable[[Callable[[object], None]], None]
    complete_reply: Callable[[list[ChatCompletionMessageParam]], str]
    synthesize: Callable[[str], bytes]
    spawn: Callable[[Callable[[], None]], _Worker] = _spawn_thread

    @classmethod
    def real(
        cls,
        api_key: str,
        config: CascadeConfig,
        *,
        audio: Iterable[bytes],
        sample_rate: int,
    ) -> CascadeDeps:
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=_stt_params(sample_rate), on_turn=on_turn)

        def complete_reply(messages: list[ChatCompletionMessageParam]) -> str:
            response = llm.complete(api_key, model=config.model, messages=messages)
            return llm.content_of(response)

        def synthesize(text: str) -> bytes:
            spec = SpeakConfig(text=text, voice=config.voice, sample_rate=TTS_SAMPLE_RATE)
            return tts_session.synthesize(api_key, spec).pcm

        return cls(run_stt=run_stt, complete_reply=complete_reply, synthesize=synthesize)


@dataclass
class CascadeSession:
    """Per-conversation state: the running history and the in-flight reply worker."""

    deps: CascadeDeps
    renderer: Renderer
    player: Player
    config: CascadeConfig
    history: list[ChatCompletionMessageParam] = field(default_factory=_new_history)
    # First leg failure (LLM/TTS). Recorded on the reply worker thread, where raising
    # would dump a thread traceback, and re-raised from the main thread to fail cleanly.
    error: CLIError | None = None
    _reply: _Worker | None = field(default=None, init=False)  # pragma: no mutate
    _stop: threading.Event = field(default_factory=threading.Event, init=False)  # pragma: no mutate

    def greet(self) -> None:
        """Speak the opening greeting (if any) and seed it into the history so the
        model has a record of its own first line."""
        greeting = self.config.greeting
        if not greeting:
            return
        self.history.append({"role": "assistant", "content": greeting})
        self.renderer.agent_transcript(greeting, interrupted=False)
        try:
            self.player.enqueue(self.deps.synthesize(greeting))
        except CLIError as exc:
            self._record_error(exc)

    def on_turn(self, event: object) -> None:
        """Handle one STT turn: reply to a finalized turn, otherwise just barge in.

        Runs on the STT reader thread. An interim turn only interrupts a playing
        reply; a finalized, formatted turn is shown and answered.
        """
        text = (getattr(event, "transcript", "") or "").strip()
        if not text:
            return
        if _is_final_turn(event):
            self.renderer.user_final(text)
            self._barge_in()
            self.history.append({"role": "user", "content": text})
            trim_history(self.history, self.config.max_history)
            self._start_reply()
        else:
            self.renderer.user_partial(text)
            self._barge_in()

    def _barge_in(self) -> None:
        """Stop a reply that is still playing: flush the queued audio and cancel the
        worker (the player flush is what silences the browser-equivalent local buffer)."""
        if self._reply is not None and self._reply.is_alive():
            self._stop.set()
            self.player.flush()
        self._join_reply()

    def _join_reply(self) -> None:
        """Wait for the current reply worker (if any) to unwind, then drop the handle."""
        worker = self._reply
        if worker is not None and worker.is_alive():
            worker.join()
        self._reply = None

    def _start_reply(self) -> None:
        self._stop.clear()
        self._reply = self.deps.spawn(self._generate_reply)

    def _generate_reply(self) -> None:
        """Stream the LLM reply, speak it sentence-by-sentence, and record what was
        actually spoken (so a barge-in still leaves the history alternating)."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.config.system_prompt},
            *self.history,
        ]
        try:
            reply = self.deps.complete_reply(messages)
        except CLIError as exc:
            self._record_error(exc)
            return
        self.renderer.reply_started()
        spoken: list[str] = []
        for sentence in split_sentences(reply):
            if self._stop.is_set():
                break
            self.renderer.agent_transcript(sentence, interrupted=False)
            try:
                pcm = self.deps.synthesize(sentence)
            except CLIError as exc:
                self._record_error(exc)
                break
            if self._stop.is_set():
                break
            self.player.enqueue(pcm)
            spoken.append(sentence)
        spoken_text = " ".join(spoken).strip()
        if spoken_text:
            self.history.append({"role": "assistant", "content": spoken_text})
            trim_history(self.history, self.config.max_history)
        self.renderer.reply_done(interrupted=self._stop.is_set())

    def _record_error(self, exc: CLIError) -> None:
        """Keep the first leg failure (to re-raise on the main thread) and warn now,
        since the worker thread can't surface an exit code itself."""
        if self.error is None:
            self.error = exc
        output.error_console.print(f"[aai.warn]agent-framework:[/aai.warn] {exc.message}")

    def shutdown(self) -> None:
        """Stop and join any in-flight reply worker (run on every exit path)."""
        self._stop.set()
        self._join_reply()


def _is_final_turn(event: object) -> bool:
    """True for a finalized, formatted end-of-turn — the cue to generate a reply."""
    return bool(getattr(event, "end_of_turn", False)) and bool(
        getattr(event, "turn_is_formatted", False)
    )


def run_cascade(
    *, renderer: Renderer, player: Player, config: CascadeConfig, deps: CascadeDeps
) -> None:
    """Run one terminal cascade conversation until STT closes or the user stops.

    Greets, then pumps STT turns through the LLM+TTS reply path. A recorded leg
    failure is re-raised here so the command exits with the right code.
    """
    session = CascadeSession(deps=deps, renderer=renderer, player=player, config=config)
    player.start()
    try:
        session.greet()
        renderer.connected()
        deps.run_stt(session.on_turn)
    finally:
        session.shutdown()
        with contextlib.suppress(Exception):
            player.close()
    if session.error is not None:
        raise session.error
