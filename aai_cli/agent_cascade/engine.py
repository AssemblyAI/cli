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

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.text import split_sentences, trim_history
from aai_cli.core import client
from aai_cli.core.errors import CLIError
from aai_cli.tts import session as tts_session
from aai_cli.tts.session import SpeakConfig
from aai_cli.ui import output

if TYPE_CHECKING:
    from assemblyai.streaming.v3 import StreamingParameters
    from openai.types.chat import ChatCompletionMessageParam

# Streaming TTS synthesizes at 24 kHz, the rate the live player is opened at.
TTS_SAMPLE_RATE = 24000

# Wall-clock backstop for one reply turn. complete_reply drives the whole deepagents graph — an
# LLM round-trip plus any tool calls — as a single blocking call with no internal deadline, so a
# stuck leg (an unresponsive gateway, a web-search tool with no timeout of its own) would hang
# the turn forever, with the worker unable to observe the stop flag. After this long we stop
# waiting and surface a timeout so the session stays usable. Generous on purpose: well above a
# normal tool-using turn, so it only fires on a genuine stall. The exact value is a tuning knob.
_REPLY_TIMEOUT_SECONDS = 60.0  # pragma: no mutate


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


def _new_history() -> list[ChatCompletionMessageParam]:
    """Typed empty-history factory (ChatCompletionMessageParam is import-time-only)."""
    return []


def _spawn_thread(target: Callable[[], None]) -> _Worker:
    """Start ``target`` on a daemon thread so a reply is generated without blocking
    the STT reader (which must stay free to detect a barge-in)."""
    thread = threading.Thread(target=target, daemon=True)  # pragma: no mutate
    thread.start()
    return thread


@dataclass
class CascadeDeps:
    """The cascade's three network legs plus its thread spawner, all injectable.

    ``CascadeDeps.real`` wires the live STT/LLM/TTS clients; tests pass fakes with
    the same shapes (and a synchronous ``spawn``) to drive the orchestration.
    """

    run_stt: Callable[[Callable[[object], None]], None]
    # complete_reply(messages, on_tool=None) -> spoken text; on_tool is fed a label per tool
    # call so the front-end can show a "Searching the web…" affordance (brain.build_completer).
    complete_reply: Callable[..., str]
    synthesize: Callable[[str], bytes]
    spawn: Callable[[Callable[[], None]], _Worker] = _spawn_thread

    @classmethod
    def real(
        cls,
        api_key: str,
        config: CascadeConfig,
        *,
        audio: Iterable[bytes],
        stt_params: StreamingParameters,
    ) -> CascadeDeps:
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=stt_params, on_turn=on_turn)

        # The LLM leg is a deepagents graph (web search / URL fetch / docs tools), not a
        # single completion, so a spoken turn can transparently use tools.
        complete_reply = brain.build_completer(api_key, config)

        def synthesize(text: str) -> bytes:
            spec = SpeakConfig(
                text=text,
                voice=config.voice,
                language=config.language,
                sample_rate=TTS_SAMPLE_RATE,
                extra=config.tts_extra,
            )
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
    # Set only while a reply is in its audible speak-and-enqueue phase (not while it's still
    # *thinking* — generating in a blocking graph call). A UI interrupt keys off this so Ctrl-C
    # can quit while the agent thinks instead of being swallowed by a no-op "interrupt".
    _speaking: threading.Event = field(
        default_factory=threading.Event, init=False
    )  # pragma: no mutate

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
        if _is_final_turn(event, format_turns=self.config.format_turns):
            self.renderer.user_final(text)
            self._barge_in()
            self.history.append({"role": "user", "content": text})
            trim_history(self.history, self.config.max_history)
            self._start_reply()
        else:
            self.renderer.user_partial(text)
            self._barge_in()

    def _silence(self, *, audible_only: bool) -> bool:
        """Cancel an in-flight reply — signal the worker and flush queued audio — and report
        whether anything was cancelled.

        The audible cases are always cancelled: the greeting (enqueued with no worker), a reply
        in its speak-and-enqueue phase (``_speaking``), and the *tail* of a reply whose worker
        has finished enqueuing but whose audio is still draining (``pending() > 0``).

        ``audible_only`` decides whether the *thinking* phase counts too. A spoken barge-in
        passes ``False`` to cancel even a reply still being generated — the user has moved on,
        so it must not speak once it lands. A UI interrupt passes ``True`` to leave thinking
        alone: there's no audio to cut and the blocking graph call can't observe the stop flag,
        so cancelling would be a no-op — and crucially, returning False there lets the TUI's
        Ctrl-C fall through to *quit* rather than be swallowed (you could otherwise never
        Ctrl-C while the agent thinks). Setting the stop flag is harmless when nothing runs (the
        next ``_start_reply`` clears it).
        """
        in_flight = self._speaking.is_set() or self.player.pending() > 0
        if not audible_only:
            in_flight = in_flight or (self._reply is not None and self._reply.is_alive())
        if in_flight:
            self._stop.set()
            self.player.flush()
        return in_flight

    def _barge_in(self) -> None:
        """Stop whatever the agent is doing (a thinking or speaking reply, the greeting, or a
        draining tail) and join — a new spoken turn supersedes it, thinking included."""
        self._silence(audible_only=False)
        self._join_reply()

    def interrupt_reply(self) -> bool:
        """Silence a *speaking* reply without waiting for it; True if one was audible.

        The UI-thread-safe counterpart to a spoken barge-in: the live TUI's Escape/Ctrl-C
        calls this to silence the agent mid-reply (or mid-greeting) without the user having to
        talk over it. Flushing the queued audio stops speech at once; a reply worker then sees
        the stop flag, unwinds on its own, and emits ``reply_done`` so the front-end returns to
        listening (the STT loop keeps running, so the next spoken turn is handled normally).
        It deliberately does *not* join the worker — a join from the UI thread would deadlock
        against the worker's own ``call_from_thread`` render hops.

        It reports False (and does nothing) while the reply is merely *thinking*, so the TUI's
        Ctrl-C falls through to quit instead of being swallowed by a no-op interrupt.
        """
        return self._silence(audible_only=True)

    def _join_reply(self) -> None:
        """Wait for the current reply worker (if any) to unwind, then drop the handle."""
        worker = self._reply
        if worker is not None and worker.is_alive():
            worker.join()
        self._reply = None

    def _start_reply(self) -> None:
        self._stop.clear()
        self._reply = self.deps.spawn(self._generate_reply)

    def _complete_within(self, messages: list[ChatCompletionMessageParam], timeout: float) -> str:
        """Run the blocking reply leg with a wall-clock backstop, returning the spoken text.

        ``complete_reply`` runs the whole deepagents graph as one uninterruptible call, so a
        stuck leg would hang the reply worker forever. Drive it on a throwaway daemon thread and
        stop waiting after ``timeout`` — raising a ``CLIError`` the caller surfaces like any
        other leg failure (inline in the transcript, then back to listening). The abandoned
        thread is a network call we can't cancel; as a daemon it dies with the process and its
        late result is discarded. A failure the leg itself raises is re-raised here unchanged.
        """
        # List holders (not closure locals) so the worker thread's result is visible here after
        # the join, and so the static checkers don't misread a nonlocal mutation as unreachable.
        replies: list[str] = []
        failures: list[CLIError] = []

        def run() -> None:
            # complete_reply (brain._run_graph) wraps every leg/tool/graph failure as a CLIError,
            # so capturing that is enough; it's re-raised on the waiting thread below.
            try:
                replies.append(self.deps.complete_reply(messages, on_tool=self.renderer.tool_call))
            except CLIError as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, daemon=True)  # pragma: no mutate
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise CLIError(
                f"the agent took longer than {timeout:.0f}s to respond and was cut off",
                error_type="agent_timeout",
            )
        if failures:
            raise failures[0]
        return replies[0]

    def _generate_reply(self) -> None:
        """Stream the LLM reply, speak it sentence-by-sentence, and record what was
        actually spoken (so a barge-in still leaves the history alternating)."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.config.system_prompt},
            *self.history,
        ]
        try:
            reply = self._complete_within(messages, _REPLY_TIMEOUT_SECONDS)
        except CLIError as exc:
            # The reply leg failed (gateway/tool/graph error, now converted to a CLIError in
            # brain._run_graph). Show it in the transcript so the turn doesn't just vanish —
            # the user sees *why* there was no answer instead of silence.
            self._record_error(exc)
            self.renderer.reply_started()
            self.renderer.agent_transcript(f"(error: {exc.message})", interrupted=False)
            self.renderer.reply_done(interrupted=False)
            return
        # The reply text is in hand — the turn moves from thinking to its audible speaking phase,
        # so a UI interrupt can now cut it (see _silence / interrupt_reply).
        self._speaking.set()
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
        # Done speaking; only a draining tail (player.pending) is still interruptible now.
        self._speaking.clear()
        self.renderer.reply_done(interrupted=self._stop.is_set())

    def _record_error(self, exc: CLIError) -> None:
        """Keep the first leg failure (to re-raise on the main thread) and warn now,
        since the worker thread can't surface an exit code itself."""
        if self.error is None:
            self.error = exc
        output.error_console.print(f"[aai.warn]agent-cascade:[/aai.warn] {exc.message}")

    def shutdown(self) -> None:
        """Stop and join any in-flight reply worker (run on every exit path)."""
        self._stop.set()
        self._join_reply()


def _is_final_turn(event: object, *, format_turns: bool) -> bool:
    """True for an end-of-turn that's the cue to generate a reply.

    With formatting on, wait for the *formatted* turn (better text for the LLM);
    with it off the server never sets ``turn_is_formatted``, so a bare end-of-turn
    is the cue — otherwise ``--no-format-turns`` would make the agent never reply.
    """
    if not bool(getattr(event, "end_of_turn", False)):
        return False
    return bool(getattr(event, "turn_is_formatted", False)) or not format_turns


def run_cascade(
    *,
    renderer: Renderer,
    player: Player,
    config: CascadeConfig,
    deps: CascadeDeps,
    on_session: Callable[[CascadeSession], None] | None = None,
) -> None:
    """Run one terminal cascade conversation until STT closes or the user stops.

    Greets, then pumps STT turns through the LLM+TTS reply path. A recorded leg
    failure is re-raised here so the command exits with the right code. ``on_session`` is
    handed the freshly built session before the conversation starts, so a front-end (the
    live TUI) can grab a handle to it — e.g. to wire a keyboard interrupt to
    :meth:`CascadeSession.interrupt_reply`.
    """
    session = CascadeSession(deps=deps, renderer=renderer, player=player, config=config)
    if on_session is not None:
        on_session(session)
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
