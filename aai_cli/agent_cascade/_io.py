"""The live cascade's I/O boundary: the render/playback protocols and injected legs.

Split out of ``engine.py`` to keep that module within the file-length gate. ``Renderer`` and
``Player`` are the surfaces the engine drives (a TUI/line renderer and a speaker);
``CascadeDeps`` bundles the three network legs plus the thread spawner so the orchestration is
unit-tested against fakes. ``engine`` re-exports all three, so importers keep using
``engine.Renderer`` / ``engine.CascadeDeps`` unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade._runtime import Worker as _Worker
from aai_cli.agent_cascade._runtime import spawn_thread as _spawn_thread
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.core import client
from aai_cli.tts import session as tts_session
from aai_cli.tts.session import SpeakConfig

if TYPE_CHECKING:
    from assemblyai.streaming.v3 import StreamingParameters

# Streaming TTS synthesizes at 24 kHz, the rate the live player is opened at.
TTS_SAMPLE_RATE = 24000


class Renderer(Protocol):
    """The conversation-rendering surface the cascade drives (AgentRenderer satisfies it)."""

    def connected(self) -> None:
        """Announce the session is live and listening."""

    def user_partial(self, text: str) -> None:
        """Show an interim user transcript."""

    def user_final(self, text: str) -> None:
        """Show a finalized user transcript."""

    def tool_call(self, label: str) -> None:
        """Show that the agent is using a tool (e.g. "Searching the web") while it thinks."""

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

    def pending(self) -> int:
        """How many unplayed samples are still queued (>0 while audio is audibly playing)."""
        ...

    def close(self) -> None:
        """Close the output stream."""


@dataclass
class CascadeDeps:
    """The cascade's three network legs plus its thread spawner, all injectable.

    ``CascadeDeps.real`` wires the live STT/LLM/TTS clients; tests pass fakes with
    the same shapes (and a synchronous ``spawn``) to drive the orchestration.
    """

    run_stt: Callable[[Callable[[object], None]], None]
    # stream_reply(messages) -> iterable of SpeechDelta/ToolNotice events (plus ApprovalPause
    # markers under --files write gating). The reply is streamed token-by-token so the engine
    # can speak each clause as it lands; a ToolNotice surfaces the "Searching the web…"
    # affordance (brain.build_streamer).
    stream_reply: Callable[
        ..., Iterable[brain.SpeechDelta | brain.ToolNotice | brain.ApprovalPause]
    ]
    # synthesize(text, sink): streaming TTS — sink is called with each PCM frame as it
    # arrives so playback starts on the first frame instead of after the whole clause.
    synthesize: Callable[[str, Callable[[bytes], None]], None]
    spawn: Callable[[Callable[[], None]], _Worker] = _spawn_thread

    @classmethod
    def real(
        cls,
        api_key: str,
        config: CascadeConfig,
        *,
        audio: Iterable[bytes],
        stt_params: StreamingParameters,
        approver: brain.Approver | None = None,
    ) -> CascadeDeps:
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=stt_params, on_turn=on_turn)

        # The LLM leg is a deepagents graph (web search / MCP tools), streamed token-by-token
        # so a spoken turn can transparently use tools and start speaking sooner. ``approver``
        # gates --files writes (None on the non-files path, where the graph never pauses).
        stream_reply = brain.build_streamer(api_key, config, approver=approver)

        def synthesize(text: str, sink: Callable[[bytes], None]) -> None:
            spec = SpeakConfig(
                text=text,
                voice=config.voice,
                language=config.language,
                sample_rate=TTS_SAMPLE_RATE,
                extra=config.tts_extra,
            )
            tts_session.synthesize(api_key, spec, on_audio=lambda chunk, _rate: sink(chunk))

        return cls(run_stt=run_stt, stream_reply=stream_reply, synthesize=synthesize)
