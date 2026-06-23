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
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade._io import CascadeDeps, Player, Renderer
from aai_cli.agent_cascade._runtime import (
    REPLY_TIMEOUT_SECONDS as _REPLY_TIMEOUT_SECONDS,
)
from aai_cli.agent_cascade._runtime import (
    Done as _Done,
)
from aai_cli.agent_cascade._runtime import (
    Failure as _Failure,
)
from aai_cli.agent_cascade._runtime import (
    ReplyEvent as _ReplyEvent,
)
from aai_cli.agent_cascade._runtime import (
    Timeout as _Timeout,
)
from aai_cli.agent_cascade._runtime import (
    Worker as _Worker,
)
from aai_cli.agent_cascade._runtime import (
    approval_deadline as _approval_deadline,
)
from aai_cli.agent_cascade._runtime import (
    detach_executor_threads_since as _detach_executor_threads_since,
)
from aai_cli.agent_cascade._runtime import (
    executor_threads as _executor_threads,
)
from aai_cli.agent_cascade._runtime import (
    final_tail as _final_tail,
)
from aai_cli.agent_cascade._runtime import (
    is_final_turn as _is_final_turn,
)
from aai_cli.agent_cascade._runtime import (
    new_history as _new_history,
)
from aai_cli.agent_cascade._runtime import (
    timeout_error as _timeout_error,
)
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.text import pop_clauses, trim_history
from aai_cli.core.errors import CLIError
from aai_cli.ui import output

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# engine is the cascade's public seam: it owns CascadeSession/run_cascade and deliberately
# re-exports the injection protocols that moved to _io (CascadeDeps/Renderer/Player), so callers
# keep importing them from here. __all__ marks the re-exports as explicit (mypy --no-implicit-reexport).
__all__ = ["CascadeDeps", "CascadeSession", "Player", "Renderer", "run_cascade"]

# A clause is flushed to TTS on a soft separator (comma/semicolon/colon) only once it is at
# least this long, so we don't synthesize a choppy two-word fragment. Pinned by a text test.
_MIN_CLAUSE_CHARS = 25


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
    # Routes a spoken approval during a --files pause (the live TUI's submit_voice_approval); None
    # on the keyboard-only/headless paths, where a spoken transcript can't answer the gate.
    on_approval_voice: Callable[[str], None] | None = None
    _reply: _Worker | None = field(default=None, init=False)  # pragma: no mutate
    _stop: threading.Event = field(default_factory=threading.Event, init=False)  # pragma: no mutate
    # Set while a --files write/run awaits approval: the next final transcript answers the gate
    # (voice) instead of starting a new turn. Armed/cleared by _consume on the ApprovalPause events.
    _awaiting_approval: threading.Event = field(
        default_factory=threading.Event,
        init=False,  # pragma: no mutate
    )
    # Set only while a reply is in its audible speak-and-enqueue phase (not while it's still
    # *thinking* — generating in a blocking graph call). A UI interrupt keys off this so Ctrl-C
    # can quit while the agent thinks instead of being swallowed by a no-op "interrupt".
    _speaking: threading.Event = field(
        default_factory=threading.Event,
        init=False,  # pragma: no mutate
    )
    # Rotates the per-tool spoken fillers across turns (fillers[_filler_index % len]) so the same
    # tool doesn't repeat one phrase. The rotation test pins the exact phrase sequence; the field's
    # `init=` is equivalent (never constructed positionally), like the siblings, hence the pragma.
    _filler_index: int = field(default=0, init=False)  # pragma: no mutate
    # reply_started fired this turn (separate from _speaking) so an empty reply still brackets it.
    # Reset per turn in _generate_reply (the killable line); init= is equivalent, hence the pragma.
    _reply_started: bool = field(default=False, init=False)  # pragma: no mutate

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
        if self._awaiting_approval.is_set():
            # A --files write/run is waiting on approval: the next *final* transcript answers the
            # gate by voice (interim partials are ignored), instead of barging in / starting a turn.
            if _is_final_turn(event, format_turns=self.config.format_turns) and (
                self.on_approval_voice is not None
            ):
                self.on_approval_voice(text)
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

        The audible cases are always cancelled: the greeting (enqueued with no worker), a reply in
        its speak-and-enqueue phase (``_speaking``), and the *tail* of a reply whose worker finished
        enqueuing but whose audio is still draining (``pending() > 0``).

        ``audible_only`` decides whether the *thinking* phase counts too. A spoken barge-in passes
        ``False`` to cancel even a reply still being generated. A UI interrupt passes ``True`` to
        leave thinking alone (no audio to cut, the blocking graph call can't see the stop flag) —
        and returning False there lets the TUI's Ctrl-C fall through to *quit* rather than be
        swallowed. Setting the stop flag is harmless when nothing runs (next ``_start_reply`` clears
        it).
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

        The UI-thread-safe counterpart to a spoken barge-in: the live TUI's Escape/Ctrl-C calls
        this to silence the agent mid-reply (or mid-greeting) without the user talking over it.
        Flushing the queued audio stops speech at once; a reply worker then sees the stop flag,
        unwinds, and emits ``reply_done`` so the front-end returns to listening (the STT loop keeps
        running). It deliberately does *not* join the worker — a join from the UI thread would
        deadlock against the worker's own ``call_from_thread`` render hops.

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
        self._reply_started = False
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
        try:
            tail = self._consume(events, before, spoken)
            # On a clean finish ``tail`` is the unspoken remainder to flush as one last clause; on a
            # cut (barge-in, TTS/leg failure, timeout) it is None and nothing more is spoken.
            if tail is not None and tail.strip():
                self._speak([tail.strip()], spoken)
        finally:
            # Always finalize the turn: clearing the gate keeps a failure between an ApprovalPause's
            # active/inactive events from wedging every later turn (and history stays alternating).
            self._set_awaiting_approval(active=False)
            self._record_spoken(spoken)
            self._speaking.clear()
            self._emit_reply_started()  # bracket reply_done even for an empty (silent) reply
            self.renderer.reply_done(interrupted=self._stop.is_set())

    def _emit_reply_started(self) -> None:
        """Fire ``reply_started`` once per turn (idempotent), so every ``reply_done`` is bracketed."""
        if not self._reply_started:
            self._reply_started = True
            self.renderer.reply_started()

    def _set_awaiting_approval(self, *, active: bool) -> None:
        """Arm/disarm the voice-approval gate: while armed, ``on_turn`` routes the next final
        transcript to the open write/run approval instead of starting a new turn."""
        if active:
            self._awaiting_approval.set()
        else:
            self._awaiting_approval.clear()

    def _consume(
        self, events: queue.Queue[_ReplyEvent], before: set[threading.Thread], spoken: list[str]
    ) -> str | None:
        """Drain the event queue, speaking each completed clause. Returns the unspoken tail to
        flush on a clean finish, or ``None`` if the turn was cut short (a barge-in stop, a TTS
        failure, or a leg failure/timeout — which also surfaces the error)."""
        deadline: float | None = time.monotonic() + _REPLY_TIMEOUT_SECONDS
        buffer = ""
        held: list[str] = []  # post-tool narration, joined once at end (O(n) vs per-delta concat)
        spoke_filler = False  # only the FIRST tool call of a turn says a spoken filler
        used_tool = False  # once a tool ran, hold text unspoken so only the final answer is read
        while True:
            item = self._next_event(events, deadline, before)
            if isinstance(item, _Timeout):
                self._surface_error(_timeout_error(), started=self._speaking.is_set())
                return None
            if isinstance(item, _Failure):
                self._surface_error(item.error, started=self._speaking.is_set())
                return None
            if isinstance(item, _Done):
                return _final_tail(buffer, held, used_tool=used_tool)
            if isinstance(item, brain.ApprovalPause):
                deadline = _approval_deadline(item)
                self._set_awaiting_approval(active=item.active)
                continue
            if isinstance(item, brain.ToolNotice):
                if not self._handle_tool_notice(item, spoke_filler=spoke_filler):
                    return None
                spoke_filler = True
                used_tool = True
                buffer = ""  # drop unspoken preamble + inter-tool narration; the answer follows
                held.clear()
                continue
            if self._stop.is_set():
                return None
            # item is a streamed SpeechDelta (every other case returned/continued above).
            tail = self._speak_delta(item, buffer, held, spoken, used_tool=used_tool)
            if tail is None:
                return None
            buffer = tail

    def _speak_delta(
        self,
        item: brain.SpeechDelta,
        buffer: str,
        held: list[str],
        spoken: list[str],
        *,
        used_tool: bool,
    ) -> str | None:
        """Fold one streamed delta into the reply, speaking any completed clauses. Pre-tool, clauses
        stream out as they land and the buffer is returned. *After* a tool call (``used_tool``) the
        verbose planning is appended to ``held`` unspoken (joined once at end of stream, so only the
        final answer is read — O(n)). Returns the new buffer, or ``None`` on a TTS failure."""
        if used_tool:
            held.append(item.text)
            return buffer
        self._mark_speaking()
        buffer += item.text
        chunks, buffer = pop_clauses(buffer, min_chars=_MIN_CLAUSE_CHARS)
        if not self._speak(chunks, spoken):
            return None
        return buffer

    def _handle_tool_notice(self, item: brain.ToolNotice, *, spoke_filler: bool) -> bool:
        """Show the tool affordance and, for the *first* tool call of a turn only, say a spoken
        filler so a hands-free turn isn't dead air. Chained tool calls (``spoke_filler``) stay
        silent. Returns False if the filler failed to synthesize (the caller aborts the turn)."""
        self.renderer.tool_call(item.label)
        if spoke_filler:
            return True
        return self._speak_filler(item.fillers)

    def _mark_speaking(self) -> None:
        """Mark the reply as audibly speaking on its first audible output — a clause or a tool
        filler. Sets ``_speaking`` (so a UI interrupt can cut it) and fires ``reply_started`` once."""
        if not self._speaking.is_set():
            self._speaking.set()
            self._emit_reply_started()

    def _speak_filler(self, fillers: tuple[str, ...]) -> bool:
        """Say a short spoken filler ("Let me check") for the first tool call of a turn, so a
        hands-free turn isn't dead air while the tool runs.

        Marks the reply speaking (the filler is the start of audible output, so a barge-in during
        it is caught), picks the next variant — rotating across turns so the same tool doesn't
        repeat one phrase — and feeds it to the player through the same ``_stop``-respecting path a
        clause uses. Unlike :meth:`_speak`, the filler is conversational glue, not part of the
        answer, so it is *never* recorded to ``spoken``/history. Returns False if synthesizing it
        failed (the caller aborts the turn, same as a clause that can't synthesize), True otherwise.
        """
        self._mark_speaking()
        text = fillers[self._filler_index % len(fillers)]
        self._filler_index += 1
        try:
            self.deps.synthesize(text, self._feed)
        except CLIError as exc:
            self._record_error(exc)
            return False
        return True

    def _next_event(
        self,
        events: queue.Queue[_ReplyEvent],
        deadline: float | None,
        before: set[threading.Thread],
    ) -> _ReplyEvent | _Timeout:
        """Block for the next streamed event until ``deadline`` (monotonic). Returns a
        :class:`_Timeout` once the deadline has passed with nothing more arriving, detaching the
        orphaned graph executor first so the abandoned producer can't wedge interpreter exit.

        ``deadline is None`` means the turn is paused awaiting human write-approval, so block
        with no timeout until the next event (the approval answer) arrives."""
        if deadline is None:
            return events.get()
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
            self._emit_reply_started()
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


def run_cascade(
    *,
    renderer: Renderer,
    player: Player,
    config: CascadeConfig,
    deps: CascadeDeps,
    on_session: Callable[[CascadeSession], None] | None = None,
    on_approval_voice: Callable[[str], None] | None = None,
) -> None:
    """Run one terminal cascade conversation until STT closes or the user stops.

    Greets, then pumps STT turns through the LLM+TTS reply path. A recorded leg
    failure is re-raised here so the command exits with the right code. ``on_session`` is
    handed the freshly built session before the conversation starts, so a front-end (the
    live TUI) can grab a handle to it — e.g. to wire a keyboard interrupt to
    :meth:`CascadeSession.interrupt_reply`.
    """
    session = CascadeSession(
        deps=deps,
        renderer=renderer,
        player=player,
        config=config,
        on_approval_voice=on_approval_voice,
    )
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
