"""Reply-streaming and interruption tests for the voice cascade (engine.py).

Covers reply generation (clause streaming, tool notices, the timeout backstop,
and leg/TTS failures) plus barge-in / interrupt / shutdown. Fixtures come from
tests/_cascade_fakes; every test runs against fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import threading

from aai_cli.agent_cascade import engine
from aai_cli.agent_cascade.brain import SpeechDelta, ToolNotice
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.core.errors import APIError, CLIError
from tests._cascade_fakes import FakeWorker, make_session
from tests._cascade_fakes import deltas as _deltas
from tests._cascade_fakes import turn as _turn

# --- reply generation --------------------------------------------------------


def test_generate_reply_pins_min_clause_chars_for_soft_separators():
    # _MIN_CLAUSE_CHARS gates SOFT separators only: a pre-comma clause whose length equals the
    # threshold flushes on the comma (two spoken clauses). If the constant were larger, that clause
    # would be held and the whole reply would speak as a single clause via the trailing period.
    # The hardcoded 25 is the expected value; a mutation (25→26) makes the 25-char clause fall
    # below the threshold, so it is not flushed at the comma and only one clause is spoken.
    assert engine._MIN_CLAUSE_CHARS == 25  # pin the exact value
    spoken = []
    text = ("a" * 24) + ", and the rest is here."  # comma-clause is exactly 25 chars -> flushes
    session, _renderer, _player = make_session(
        stream_reply=_deltas(text),
        synthesize=lambda t, sink: spoken.append(t) or sink(b""),
    )
    session._generate_reply()
    assert len(spoken) == 2
    assert spoken[0].endswith(",")  # the soft clause flushed at the comma because len >= 25


def test_generate_reply_speaks_each_clause_as_it_streams():
    spoken = []
    session, renderer, player = make_session(
        stream_reply=_deltas("One. ", "Two! ", "Three?"),
        synthesize=lambda text, sink: spoken.append(text) or sink(text.encode()),
    )
    session._generate_reply()
    assert spoken == ["One.", "Two!", "Three?"]
    assert player.enqueued == [b"One.", b"Two!", b"Three?"]
    assert ("reply_started",) in renderer.calls
    assert ("agent_transcript", "One.", False) in renderer.calls
    assert session.history[-1] == {"role": "assistant", "content": "One. Two! Three?"}
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_forwards_tool_notice_and_drops_unspoken_preamble():
    # A ToolNotice surfaces the affordance AND clears any buffered-but-unspoken text, so a
    # half-streamed preamble before a tool call is never spoken.
    spoken = []

    def stream(messages):
        yield SpeechDelta("Let me check")  # incomplete clause, not yet flushed
        yield ToolNotice("Searching the web", ("Searching now.",))
        yield SpeechDelta("It is sunny today.")

    session, renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert ("tool_call", "Searching the web") in renderer.calls
    # The unspoken preamble is dropped; the spoken filler takes its place before the answer.
    assert spoken == ["Searching now.", "It is sunny today."]
    assert session.history[-1] == {"role": "assistant", "content": "It is sunny today."}


def test_generate_reply_speaks_a_filler_on_the_first_tool_call():
    # While a tool runs the agent says a spoken filler so a hands-free turn isn't dead air; it is
    # synthesized BEFORE the answer clauses (which only land after the tool returns).
    spoken = []

    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.", "Checking now."))
        yield SpeechDelta("It is sunny today.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["Let me check the weather.", "It is sunny today."]


def test_generate_reply_discards_planning_text_emitted_between_tool_calls():
    # A deepagents turn interleaves verbose planning ("SESSION INTENT / NEXT STEPS") with tool
    # calls; only the FINAL answer (the text after the last tool call) is spoken and recorded —
    # the intermediate planning is held unspoken and discarded at the next tool call.
    spoken = []

    def stream(messages):
        yield ToolNotice("Searching the web", ("Searching now.",))
        yield SpeechDelta("SESSION INTENT: the user asked for the news. ")  # planning, post-tool
        yield SpeechDelta("NEXT STEPS: read a few more pages. ")  # more planning
        yield ToolNotice("Reading the page", ("Reading now.",))  # discards the buffered planning
        yield SpeechDelta("Here is the news: AI is booming.")  # the real answer, after last tool

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["Searching now.", "Here is the news: AI is booming."]
    assert session.history[-1] == {
        "role": "assistant",
        "content": "Here is the news: AI is booming.",
    }


def test_generate_reply_speaks_only_one_filler_per_turn():
    # Only the first tool call of a turn speaks; chained tool calls stay silent so a multi-tool
    # turn doesn't get chatty.
    spoken = []

    def stream(messages):
        yield ToolNotice("Searching the web", ("look one", "look two"))
        yield ToolNotice("Reading the page", ("read one", "read two"))
        yield SpeechDelta("Done.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["look one", "Done."]  # the second tool call spoke no filler


def test_generate_reply_rotates_fillers_across_turns():
    # A per-session counter rotates variants deterministically, so the same tool across three
    # turns says each variant in turn rather than repeating one phrase.
    spoken = []
    variants = ("first.", "second.", "third.")

    def stream(messages):
        yield ToolNotice("Checking the weather", variants)
        yield SpeechDelta("ok.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    session._generate_reply()
    session._generate_reply()
    assert [s for s in spoken if s != "ok."] == ["first.", "second.", "third."]


def test_generate_reply_does_not_record_the_filler_in_history():
    # The filler is conversational glue, not part of the answer, so history stays a clean
    # alternating record of the real reply only.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("It is sunny today.")

    session, _renderer, _player = make_session(
        stream_reply=stream, synthesize=lambda text, sink: sink(b"")
    )
    session._generate_reply()
    assert session.history[-1] == {"role": "assistant", "content": "It is sunny today."}


def test_generate_reply_marks_started_when_a_turn_opens_with_a_tool_call():
    # The filler is the start of audible output, so _speaking is set and reply_started fires
    # before it's synthesized (so the voice bar shows speaking and a barge-in mid-filler is caught).
    observed = []

    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check.",))
        yield SpeechDelta("ok.")

    session, renderer, _player = make_session(stream_reply=stream)
    session.deps.synthesize = lambda text, sink: (
        observed.append((text, session._speaking.is_set())) or sink(b"")
    )
    session._generate_reply()
    assert ("reply_started",) in renderer.calls
    assert observed[0] == ("Let me check.", True)  # speaking set before the filler is synthesized


def test_barge_in_during_a_filler_drops_its_remaining_audio():
    # A barge-in mid-filler sets _stop; _feed must drop the in-flight frames just like a clause,
    # so nothing of the filler keeps playing after the user talks over it.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("answer")

    session, _renderer, player = make_session(stream_reply=stream)

    def synthesize(text, sink):
        session._stop.set()  # the user barges in mid-filler
        sink(b"frame")

    session.deps.synthesize = synthesize
    session._generate_reply()
    assert player.enqueued == []  # _feed dropped the frame once _stop was set


def test_generate_reply_aborts_when_the_filler_fails_to_synthesize():
    # A filler that can't be synthesized is the same failure mode as a clause that can't: the
    # error is recorded and the turn ends cleanly without speaking the answer.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("the answer")

    spoken = []

    def synthesize(text, sink):
        if text == "the answer":
            spoken.append(text)
            return
        raise APIError("tts boom")

    session, _renderer, _player = make_session(stream_reply=stream, synthesize=synthesize)
    session._generate_reply()
    assert spoken == []  # the answer was never reached
    assert session.error is not None
    assert "tts boom" in session.error.message


def test_generate_reply_speaks_a_generic_filler_for_an_unknown_tool():
    # A tool with no entry in _TOOL_FILLERS still speaks — the notice carries the generic
    # fallback tuple (mirrors _tool_label's "Using {name}" fallback).
    from aai_cli.agent_cascade import brain

    spoken = []

    def stream(messages):
        yield ToolNotice("Using mcp_thing", brain._tool_fillers("mcp_thing"))
        yield SpeechDelta("done.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == [brain._GENERIC_FILLERS[0], "done."]


def test_generate_reply_marks_speaking_on_first_delta_then_clears():
    observed = []
    session, _renderer, _player = make_session(stream_reply=_deltas("Hi. ", "Yes."))
    session.deps.synthesize = lambda text, sink: observed.append(session._speaking.is_set())
    session._generate_reply()
    assert observed == [True, True]
    assert not session._speaking.is_set()


def test_generate_reply_threads_system_prompt_and_history():
    captured = {}

    def capture(messages):
        captured["messages"] = messages
        return [SpeechDelta("Ok.")]

    session, _renderer, _player = make_session(
        stream_reply=capture, config=CascadeConfig(system_prompt="be terse")
    )
    session.history.append({"role": "user", "content": "prior"})
    session._generate_reply()
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert {"role": "user", "content": "prior"} in captured["messages"]


def test_generate_reply_trims_history_window():
    session, _renderer, _player = make_session(
        stream_reply=_deltas("a. b."), config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "user", "content": "hi"})
    session._generate_reply()
    assert session.history == [{"role": "assistant", "content": "a. b."}]


def test_on_turn_trims_history_window():
    session, _renderer, _player = make_session(
        stream_reply=_deltas(""), config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "assistant", "content": "old"})
    session.on_turn(_turn("newest"))
    assert session.history == [{"role": "user", "content": "newest"}]


def test_generate_reply_stop_during_a_clause_drops_it_from_the_record():
    # A barge-in lands *while* "Two." is synthesizing: its audio is flushed and the clause is NOT
    # recorded as spoken (the user never heard it whole), so only the finished "One." survives —
    # the post-synthesis stop check is what keeps the half-spoken clause out of the history.
    def synth(text, sink):
        if text == "Two.":
            session._stop.set()  # barge-in mid-clause: its frames are dropped by _feed
        sink(text.encode())

    session, renderer, player = make_session(stream_reply=_deltas("One. Two. Three."))
    session.deps.synthesize = synth
    session._generate_reply()
    assert player.enqueued == [b"One."]  # Two.'s frames are dropped once the stop lands
    assert session.history[-1] == {"role": "assistant", "content": "One."}
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_flushes_the_unterminated_tail_at_end_of_stream():
    # A reply that never ends on a terminator still gets spoken: the trailing buffer is
    # flushed as one final clause when the stream finishes.
    spoken = []
    session, _renderer, player = make_session(
        stream_reply=_deltas("no terminator here"),
        synthesize=lambda text, sink: spoken.append(text) or sink(text.encode()),
    )
    session._generate_reply()
    assert spoken == ["no terminator here"]
    assert player.enqueued == [b"no terminator here"]
    assert session.history[-1] == {"role": "assistant", "content": "no terminator here"}


def test_generate_reply_leg_failure_after_speaking_keeps_the_spoken_text():
    # A leg error that arrives *after* a clause was spoken is recorded but not shown inline
    # (the spoken text already explains the turn); the spoken part stays in the history.
    def stream(messages):
        yield SpeechDelta("First clause. ")
        raise APIError("gateway died midway")

    session, renderer, player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: sink(text.encode()),
    )
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert player.enqueued == [b"First clause."]
    assert session.history[-1] == {"role": "assistant", "content": "First clause."}
    # The error is NOT surfaced inline once speech has started (no "(error: ...)" line).
    assert not any(c[0] == "agent_transcript" and "(error:" in c[1] for c in renderer.calls)
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_stop_before_first_clause_speaks_nothing():
    session, renderer, player = make_session(stream_reply=_deltas("One. Two."))
    session._stop.set()
    session._generate_reply()
    assert player.enqueued == []
    assert all(item.get("role") != "assistant" for item in session.history)
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_times_out_via_the_backstop(monkeypatch):
    release = threading.Event()

    def hang(messages):
        release.wait(timeout=2.0)  # self-releases so no mutated deadline can wedge the suite
        yield SpeechDelta("late")

    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.05)
    session, renderer, player = make_session(stream_reply=hang)
    try:
        session._generate_reply()
        assert isinstance(session.error, CLIError)
        assert session.error.error_type == "agent_timeout"
        assert any(c[0] == "agent_transcript" and "longer than" in c[1] for c in renderer.calls)
        assert ("reply_done", False) in renderer.calls
        assert player.enqueued == []
    finally:
        release.set()


def test_generate_reply_with_an_already_elapsed_deadline_times_out_at_once(monkeypatch):
    # A non-positive remaining budget (the deadline is already in the past on the first wait)
    # surfaces the timeout immediately without ever blocking on the event queue.
    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.0)
    session, renderer, player = make_session(stream_reply=_deltas("would have spoken."))
    session._generate_reply()
    assert isinstance(session.error, CLIError)
    assert session.error.error_type == "agent_timeout"
    assert player.enqueued == []  # nothing is ever pulled off the queue
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_detaches_the_orphaned_executor_on_timeout(monkeypatch):
    # Regression: the streamed graph drives each node through a langchain ThreadPoolExecutor.
    # A timed-out turn abandons the producer with that worker still blocked on the leg, and
    # concurrent.futures joins every executor worker at interpreter exit — a blocked one wedges
    # shutdown. _generate_reply's timeout path must unregister that orphan.
    import concurrent.futures.thread as cf_thread
    from concurrent.futures import ThreadPoolExecutor

    from aai_cli.agent_cascade.brain import SpeechDelta

    release = threading.Event()
    executors: list[ThreadPoolExecutor] = []

    def hang(messages):
        executor = ThreadPoolExecutor(max_workers=1)
        executors.append(executor)
        executor.submit(lambda: release.wait(timeout=2.0)).result()
        yield SpeechDelta("late")

    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.2)
    session, _renderer, _player = make_session(stream_reply=hang)
    before = set(cf_thread._threads_queues)
    try:
        session._generate_reply()
        assert isinstance(session.error, CLIError)
        assert session.error.error_type == "agent_timeout"
        assert set(cf_thread._threads_queues) - before == set()
    finally:
        release.set()
        for executor in executors:
            executor.shutdown(wait=True)


def test_generate_reply_llm_failure_is_recorded_and_surfaced():
    def boom(messages):
        raise APIError("gateway down")

    session, renderer, player = make_session(stream_reply=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert ("agent_transcript", "(error: gateway down)", False) in renderer.calls
    assert ("reply_done", False) in renderer.calls
    assert player.enqueued == []


def test_generate_reply_tts_failure_midway_is_recorded():
    def boom(text, sink):
        raise APIError("tts down")

    session, renderer, player = make_session(stream_reply=_deltas("Hi."), synthesize=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert player.enqueued == []
    assert ("reply_started",) in renderer.calls
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_tts_failure_aborts_the_rest_of_the_turn():
    # A TTS failure cuts the turn: the leg is down, so a *later* streamed delta ("After.") is
    # never synthesized — the turn aborts on the failure rather than speaking on.
    spoken = []

    def stream(messages):
        yield SpeechDelta("Boom. ")
        yield SpeechDelta("After.")

    def synth(text, sink):
        if text == "Boom.":
            raise APIError("tts down")
        spoken.append(text)
        sink(text.encode())

    session, _renderer, player = make_session(stream_reply=stream)
    session.deps.synthesize = synth
    session._generate_reply()
    assert spoken == []  # After. is never reached once Boom. fails the leg
    assert player.enqueued == []
    assert all(item.get("role") != "assistant" for item in session.history)


def test_generate_reply_succeeds_within_a_short_deadline(monkeypatch):
    # A reply that lands inside a tight (sub-second) deadline is spoken normally — the deadline
    # only fires on a genuine stall, not on every turn.
    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.5)
    session, _renderer, player = make_session(stream_reply=_deltas("Quick reply."))
    session._generate_reply()
    assert session.error is None
    assert player.enqueued == [b"pcm:Quick reply."]
    assert session.history[-1] == {"role": "assistant", "content": "Quick reply."}


def test_record_error_keeps_first_and_warns(monkeypatch):
    printed = []
    monkeypatch.setattr(engine.output.error_console, "print", lambda msg: printed.append(msg))
    session, _renderer, _player = make_session()
    session._record_error(APIError("first"))
    session._record_error(APIError("second"))
    assert isinstance(session.error, APIError)
    assert session.error.message == "first"
    assert any("first" in str(msg) for msg in printed)


# --- barge-in / shutdown -----------------------------------------------------


def test_barge_in_cancels_and_flushes_live_worker():
    # A new spoken turn supersedes a reply that is still *thinking* (alive, not yet speaking):
    # unlike a UI interrupt, a barge-in must cancel it so it never speaks over the new turn.
    session, _renderer, player = make_session()
    worker = FakeWorker(alive=True)
    session._reply = worker
    session._barge_in()
    assert session._stop.is_set()
    assert player.flushed == 1
    assert worker.joined == 1
    assert session._reply is None


def test_barge_in_without_a_live_worker_does_not_flush():
    # No worker, or one that already finished: nothing to cancel, so no flush.
    session, _renderer, player = make_session()
    session._barge_in()  # no worker
    session._reply = FakeWorker(alive=False)
    session._barge_in()  # finished worker
    assert player.flushed == 0
    assert session._reply is None


def test_interrupt_reply_signals_stop_and_flushes_without_joining():
    # Live TUI Escape/Ctrl-C silences a *speaking* reply: stop flag + flush, but NO join.
    session, _renderer, player = make_session()
    worker = FakeWorker(alive=True)
    session._reply = worker
    session._speaking.set()  # the reply has reached its speak-and-enqueue phase
    assert session.interrupt_reply() is True
    assert session._stop.is_set()
    assert player.flushed == 1
    assert worker.joined == 0  # not joined — the worker unwinds on its own
    assert session._reply is worker  # still tracked; the next turn's barge-in joins it


def test_interrupt_reply_while_thinking_returns_false_so_ctrl_c_can_quit():
    # The reply worker is alive but still *thinking* (generating, no audio yet): there's nothing
    # audible to cut and the blocking graph can't observe the stop flag, so a UI interrupt is a
    # no-op. It must report False (not the bare is_alive() True) so the TUI's Ctrl-C falls
    # through to quit instead of being swallowed — otherwise you can't Ctrl-C while it thinks.
    session, _renderer, player = make_session()
    session._reply = FakeWorker(alive=True)  # thinking: alive, but _speaking is not set
    assert session.interrupt_reply() is False
    assert not session._stop.is_set()  # nothing cancelled — the keypress is free to quit
    assert player.flushed == 0


def test_interrupt_reply_is_a_noop_when_nothing_is_playing():
    # No worker, or one that already finished: nothing to stop, so no flush and no stop flag.
    session, _renderer, player = make_session()
    assert session.interrupt_reply() is False  # no worker
    session._reply = FakeWorker(alive=False)
    assert session.interrupt_reply() is False  # finished worker
    assert player.flushed == 0
    assert not session._stop.is_set()


def test_interrupt_reply_silences_the_greeting_with_no_worker():
    # The greeting is enqueued with no reply worker; Escape/Ctrl-C must still cut it. With audio
    # queued (pending>0) the interrupt flushes the player and reports that it silenced something,
    # so the live TUI interrupts the greeting instead of (for Ctrl-C) quitting the session.
    session, _renderer, player = make_session()
    player.pending_samples = 1  # even a single queued sample (>0) means sound is still playing
    assert session.interrupt_reply() is True
    assert player.flushed == 1
    assert player.pending() == 0  # the queued greeting was dropped


def test_barge_in_silences_a_draining_reply_tail_after_the_worker_exits():
    # The reply worker enqueues every sentence then exits, but the audio keeps draining. A new
    # spoken turn in that window must still cut the tail — a bare is_alive() check would miss it.
    session, _renderer, player = make_session()
    session._reply = FakeWorker(alive=False)  # worker finished enqueuing
    player.pending_samples = 9600  # ...but its audio is still playing
    session._barge_in()
    assert session._stop.is_set()
    assert player.flushed == 1
    assert session._reply is None


def test_shutdown_joins_live_worker():
    session, _renderer, _player = make_session()
    worker = FakeWorker(alive=True)
    session._reply = worker
    session.shutdown()
    assert session._stop.is_set()
    assert worker.joined == 1
    assert session._reply is None


def test_shutdown_without_worker_is_safe():
    session, _renderer, _player = make_session()
    session.shutdown()  # no worker spawned
    assert session._reply is None
