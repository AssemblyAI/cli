"""The terminal voice cascade: Streaming STT -> LLM Gateway -> streaming TTS.

``run_cascade`` greets the user, then drives a live conversation by reading STT
turns and, for each finalized turn, streaming an LLM reply out through TTS
sentence-by-sentence. A new turn barges in on a reply that is still playing.

All three network legs are injected through ``CascadeDeps`` (the same seam
``aai_cli/tts/session.py`` uses), so the orchestration is unit-tested against
fakes with no sockets, microphone, or speaker.
"""

from __future__ import annotations

import concurrent.futures.thread as cf_thread
import contextlib
import queue
import threading
import time
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.text import pop_clauses, trim_history
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

# Wall-clock backstop for one reply turn. The reply is streamed on a throwaway producer
# thread feeding a queue; a stalled gateway can block inside a token read the worker can't
# observe, so the consumer's queue.get is bounded by a monotonic deadline. After this long
# we stop waiting and surface a timeout so the session stays usable. Generous on purpose.
_REPLY_TIMEOUT_SECONDS = 60.0  # pragma: no mutate

# A clause is flushed to TTS on a soft separator (comma/semicolon/colon) only once it is at
# least this long, so we don't synthesize a choppy two-word fragment. Pinned by a text test.
_MIN_CLAUSE_CHARS = 25


@dataclass(frozen=True)
class _Done:
    """Producer sentinel: the reply stream finished normally."""


@dataclass(frozen=True)
class _Failure:
    """Producer sentinel: the reply leg raised a (clean) CLIError."""

    error: CLIError


@dataclass(frozen=True)
class _Timeout:
    """Consumer sentinel: the wall-clock deadline elapsed before the next event arrived."""


# What the producer thread puts on the consumer's queue: a speech/tool event from the
# streaming leg, or a terminal sentinel (clean finish / clean failure).
type _ReplyEvent = brain.SpeechDelta | brain.ToolNotice | _Done | _Failure


def _timeout_error() -> CLIError:
    """The backstop error raised when a reply overruns the wall-clock deadline."""
    return CLIError(
        f"the agent took longer than {_REPLY_TIMEOUT_SECONDS:.0f}s to respond and was cut off",
        error_type="agent_timeout",
    )


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


def _executor_threads() -> set[threading.Thread]:
    """A snapshot of every live ThreadPoolExecutor worker concurrent.futures tracks for its
    interpreter-exit join. Empty if a future Python drops the internal registry."""
    return set(getattr(cf_thread, "_threads_queues", ()))


def _detach_executor_threads_since(before: set[threading.Thread]) -> None:
    """Drop executor workers spawned since ``before`` from concurrent.futures' exit-join list,
    so an abandoned (timed-out) graph leg can't wedge process exit.

    ``complete_reply`` runs the deepagents graph, which drives each node through a langchain
    ``ThreadPoolExecutor``. Abandoning a timed-out call leaves that executor's worker blocked on
    the network leg, and concurrent.futures registers an interpreter-exit hook (``_python_exit``)
    that joins *every* executor worker unconditionally — even daemons — by putting a shutdown
    sentinel on its queue and waiting. A worker mid-call never reads that sentinel, so the join
    (and the whole process exit) hangs until the user Ctrl-Cs — the threading-shutdown traceback
    this prevents. The worker was created on our own daemon thread so it inherits ``daemon=True``;
    once it's off this registry neither ``_python_exit`` nor ``threading._shutdown`` waits on it,
    and the orphaned network call dies with the process as a daemon should. Best-effort: a future
    Python that renames the internals simply skips the detach (regressing to the old hang, not
    crashing). The diff is scoped to threads that appeared during the call, so a co-running
    executor elsewhere keeps its normal exit-time join.
    """
    registry = getattr(cf_thread, "_threads_queues", None)
    if registry is None:
        return
    # Mutate under the same lock concurrent.futures holds for the registry, so a concurrent
    # submit (or _python_exit itself) never sees a torn dict.
    with getattr(cf_thread, "_global_shutdown_lock", contextlib.nullcontext()):
        for thread in _executor_threads() - before:
            registry.pop(thread, None)


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
    # stream_reply(messages) -> iterable of SpeechDelta/ToolNotice events. The reply is
    # streamed token-by-token so the engine can speak each clause as it lands; a ToolNotice
    # surfaces the "Searching the web…" affordance (brain.build_streamer).
    stream_reply: Callable[..., Iterable[brain.SpeechDelta | brain.ToolNotice]]
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
    ) -> CascadeDeps:
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=stt_params, on_turn=on_turn)

        # The LLM leg is a deepagents graph (web search / MCP tools), streamed token-by-token
        # so a spoken turn can transparently use tools and start speaking sooner.
        stream_reply = brain.build_streamer(api_key, config)

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
            self.deps.synthesize(greeting, self.player.enqueue)
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

    def _generate_reply(self) -> None:
        """Stream the LLM reply, speak each clause as it lands, and record what was spoken
        (so a barge-in still leaves the history alternating)."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.config.system_prompt},
            *self.history,
        ]
        events: queue.Queue[_ReplyEvent] = queue.Queue()
        before = _executor_threads()

        def produce() -> None:
            self._pump(messages, events)

        producer = threading.Thread(target=produce, daemon=True)  # pragma: no mutate
        producer.start()
        spoken: list[str] = []
        tail = self._consume(events, before, spoken)
        # On a clean finish ``tail`` is the unspoken remainder to flush as one last clause; on
        # any cut (barge-in, TTS/leg failure, timeout) it is None and nothing more is spoken.
        if tail is not None and tail.strip():
            self._speak([tail.strip()], spoken)
        # Always record what was spoken — even after a mid-turn leg failure — so the history
        # stays alternating and the next turn has the partial answer as context.
        self._record_spoken(spoken)
        self._speaking.clear()
        self.renderer.reply_done(interrupted=self._stop.is_set())

    def _consume(
        self, events: queue.Queue[_ReplyEvent], before: set[threading.Thread], spoken: list[str]
    ) -> str | None:
        """Drain the event queue, speaking each completed clause. Returns the unspoken tail to
        flush on a clean finish, or ``None`` if the turn was cut short (a barge-in stop, a TTS
        failure, or a leg failure/timeout — which also surfaces the error)."""
        deadline = time.monotonic() + _REPLY_TIMEOUT_SECONDS
        buffer = ""
        started = False
        while True:
            item = self._next_event(events, deadline, before)
            if isinstance(item, _Timeout):
                self._surface_error(_timeout_error(), started=started)
                return None
            if isinstance(item, _Failure):
                self._surface_error(item.error, started=started)
                return None
            if isinstance(item, _Done):
                return buffer
            if isinstance(item, brain.ToolNotice):
                self.renderer.tool_call(item.label)
                buffer = ""  # drop any unspoken preamble — the answer comes after the tool
                continue
            if self._stop.is_set():
                return None
            if not started:
                self._speaking.set()
                self.renderer.reply_started()
                started = True
            buffer += item.text
            chunks, buffer = pop_clauses(buffer, min_chars=_MIN_CLAUSE_CHARS)
            if not self._speak(chunks, spoken):
                return None

    def _next_event(
        self, events: queue.Queue[_ReplyEvent], deadline: float, before: set[threading.Thread]
    ) -> _ReplyEvent | _Timeout:
        """Block for the next streamed event until ``deadline`` (monotonic). Returns a
        :class:`_Timeout` once the deadline has passed with nothing more arriving, detaching the
        orphaned graph executor first so the abandoned producer can't wedge interpreter exit."""
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                return events.get(timeout=remaining)
            except queue.Empty:
                pass
        # The producer is still blocked inside the graph's langchain ThreadPoolExecutor; detach
        # that orphaned worker so it can't wedge interpreter exit before we surface the timeout.
        _detach_executor_threads_since(before)
        return _Timeout()

    def _pump(
        self, messages: list[ChatCompletionMessageParam], events: queue.Queue[_ReplyEvent]
    ) -> None:
        """Drive the streaming reply leg on a throwaway thread, forwarding events to the
        queue and ending with a _Done (or _Failure on a clean leg error)."""
        try:
            for event in self.deps.stream_reply(messages):
                events.put(event)
            events.put(_Done())
        except CLIError as exc:
            events.put(_Failure(exc))

    def _speak(self, chunks: list[str], spoken: list[str]) -> bool:
        """Render and synthesize each clause, feeding frames to the player. Returns False when a
        TTS failure cut the turn (the caller aborts); True otherwise. A barge-in stop mid-clause
        stops appending (the half-heard clause is dropped from the record) and the consumer's own
        stop check ends the turn on the next event."""
        for chunk in chunks:
            self.renderer.agent_transcript(chunk, interrupted=False)
            try:
                self.deps.synthesize(chunk, self._feed)
            except CLIError as exc:
                self._record_error(exc)
                return False
            if self._stop.is_set():
                break  # barge-in landed: leave this clause unrecorded, let _consume abort
            spoken.append(chunk)
        return True

    def _feed(self, pcm: bytes) -> None:
        """Enqueue one synthesized PCM frame, unless a barge-in has already landed (then the
        remaining frames of the in-flight clause are dropped)."""
        if not self._stop.is_set():
            self.player.enqueue(pcm)

    def _record_spoken(self, spoken: list[str]) -> None:
        """Append what was actually spoken to the history (kept alternating after a barge-in)."""
        spoken_text = " ".join(spoken).strip()
        if spoken_text:
            self.history.append({"role": "assistant", "content": spoken_text})
            trim_history(self.history, self.config.max_history)

    def _surface_error(self, exc: CLIError, *, started: bool) -> None:
        """Record a reply-leg failure (LLM/timeout). Before any audio, the error is also shown
        inline in the transcript so the turn doesn't vanish; mid-speech it is only recorded (the
        spoken text already explains the turn). The caller still finalizes the turn."""
        self._record_error(exc)
        if not started:
            self.renderer.reply_started()
            self.renderer.agent_transcript(f"(error: {exc.message})", interrupted=False)

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
