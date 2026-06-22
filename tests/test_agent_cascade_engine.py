"""Orchestration tests for the terminal voice cascade (aai_cli.agent_cascade.engine).

The cascade's three network legs and its thread spawner are injected through
CascadeDeps, so every test here runs against fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import threading
import types

import pytest

from aai_cli.agent_cascade import engine
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.engine import CascadeDeps, CascadeSession, run_cascade
from aai_cli.core.errors import APIError, CLIError
from tests._cascade_fakes import FakePlayer, FakeRenderer, FakeWorker, make_session
from tests._cascade_fakes import sync_spawn as _sync_spawn
from tests._cascade_fakes import turn as _turn

# --- greeting ----------------------------------------------------------------


def test_greet_speaks_and_seeds_history():
    session, renderer, player = make_session()
    session.greet()
    assert session.history == [{"role": "assistant", "content": session.config.greeting}]
    assert ("agent_transcript", session.config.greeting, False) in renderer.calls
    assert player.enqueued == [b"pcm:" + session.config.greeting.encode()]


def test_greet_empty_greeting_is_silent():
    session, renderer, player = make_session(config=CascadeConfig(greeting=""))
    session.greet()
    assert session.history == []
    assert renderer.calls == []
    assert player.enqueued == []


def test_greet_records_tts_failure():
    def boom(text):
        raise APIError("tts down")

    session, _renderer, player = make_session(synthesize=boom)
    session.greet()
    assert isinstance(session.error, APIError)
    assert session.error.message == "tts down"
    assert player.enqueued == []  # the failed greeting enqueued nothing


# --- turn dispatch -----------------------------------------------------------


def test_on_turn_blank_transcript_ignored():
    session, renderer, _player = make_session()
    session.on_turn(_turn("   "))
    assert renderer.calls == []
    assert session.history == []


def test_on_turn_final_renders_and_replies():
    session, renderer, player = make_session(complete_reply=lambda m, on_tool=None: "Sure thing.")
    session.on_turn(_turn("what time is it"))
    assert ("user_final", "what time is it") in renderer.calls
    assert {"role": "user", "content": "what time is it"} in session.history
    assert {"role": "assistant", "content": "Sure thing."} in session.history
    assert player.enqueued == [b"pcm:Sure thing."]
    assert ("reply_done", False) in renderer.calls


def test_reply_forwards_tool_calls_to_the_renderer():
    # The reply worker hands complete_reply an on_tool sink; a tool call it makes surfaces on
    # the renderer, so the live UI can show a "Searching the web…" affordance mid-turn.
    def reply(messages, on_tool):
        on_tool("Searching the web")
        return "Found it."

    session, renderer, _player = make_session(complete_reply=reply)
    session.on_turn(_turn("what's the news"))
    assert ("tool_call", "Searching the web") in renderer.calls


def test_on_turn_interim_shows_partial_and_does_not_reply():
    replies = []
    session, renderer, _player = make_session(
        complete_reply=lambda m, on_tool=None: replies.append(m) or "x"
    )
    session.on_turn(_turn("partial words", end_of_turn=False))
    assert ("user_partial", "partial words") in renderer.calls
    assert replies == []  # no reply generated for an interim turn
    assert session.history == []


def test_on_turn_interim_barges_in_on_live_reply():
    session, _renderer, player = make_session()
    session._reply = FakeWorker(alive=True)
    session.on_turn(_turn("uh", end_of_turn=False))
    assert player.flushed == 1
    assert session._reply is None


# --- reply generation --------------------------------------------------------


def test_generate_reply_speaks_each_sentence():
    spoken = []
    session, renderer, player = make_session(
        complete_reply=lambda m, on_tool=None: "One. Two! Three?",
        synthesize=lambda text: spoken.append(text) or text.encode(),
    )
    session._generate_reply()
    assert spoken == ["One.", "Two!", "Three?"]
    assert player.enqueued == [b"One.", b"Two!", b"Three?"]
    assert ("reply_started",) in renderer.calls
    assert ("agent_transcript", "One.", False) in renderer.calls
    assert session.history[-1] == {"role": "assistant", "content": "One. Two! Three?"}
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_marks_speaking_during_playback_then_clears():
    # The reply is "speaking" only while it enqueues sentences — so a UI interrupt cuts it then,
    # but the prior thinking phase (and the idle window after) is not interruptible. The flag is
    # set before the first sentence and cleared once the turn is done.
    observed = []
    session, _renderer, _player = make_session(complete_reply=lambda m, on_tool=None: "Hi. Yes.")
    session.deps.synthesize = lambda text: observed.append(session._speaking.is_set()) or b""
    session._generate_reply()
    assert observed == [True, True]  # speaking while each sentence plays
    assert not session._speaking.is_set()  # cleared once the reply is done


def test_generate_reply_threads_system_prompt_and_history():
    captured = {}

    def capture(messages, on_tool=None):
        captured["messages"] = messages
        return "Ok."

    session, _renderer, _player = make_session(
        complete_reply=capture, config=CascadeConfig(system_prompt="be terse")
    )
    session.history.append({"role": "user", "content": "prior"})
    session._generate_reply()
    assert captured["messages"][0] == {"role": "system", "content": "be terse"}
    assert {"role": "user", "content": "prior"} in captured["messages"]


def test_generate_reply_trims_history_window():
    session, _renderer, _player = make_session(
        complete_reply=lambda m, on_tool=None: "a. b.", config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "user", "content": "hi"})
    session._generate_reply()
    # user + assistant would be 2; the window caps it to the most recent 1.
    assert session.history == [{"role": "assistant", "content": "a. b."}]


def test_on_turn_trims_history_window():
    # An empty reply adds no assistant turn, so only on_turn's own trim caps the list.
    session, _renderer, _player = make_session(
        complete_reply=lambda m, on_tool=None: "", config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "assistant", "content": "old"})
    session.on_turn(_turn("newest"))
    assert session.history == [{"role": "user", "content": "newest"}]


def test_generate_reply_stop_after_first_sentence_records_partial():
    def synth(text):
        if text == "Two.":
            session._stop.set()
        return text.encode()

    session, renderer, player = make_session(
        complete_reply=lambda m, on_tool=None: "One. Two. Three."
    )
    session.deps.synthesize = synth
    session._generate_reply()
    # Only the first sentence finished enqueuing before the barge-in stop landed.
    assert player.enqueued == [b"One."]
    assert session.history[-1] == {"role": "assistant", "content": "One."}
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_stop_before_first_sentence_speaks_nothing():
    session, renderer, player = make_session(complete_reply=lambda m, on_tool=None: "One. Two.")
    session._stop.set()
    session._generate_reply()
    assert player.enqueued == []
    # nothing spoken -> no assistant turn recorded
    assert all(item.get("role") != "assistant" for item in session.history)
    assert ("reply_done", True) in renderer.calls


def test_complete_within_returns_reply_before_the_deadline():
    # The fast path: the leg finishes well inside the deadline, so its text is returned as-is.
    session, _renderer, _player = make_session(complete_reply=lambda m, on_tool=None: "quick")
    assert session._complete_within([{"role": "user", "content": "hi"}], timeout=5.0) == "quick"


def test_complete_within_raises_a_timeout_when_the_leg_overruns_the_deadline():
    # The backstop: a leg that blocks past the deadline is cut off with an agent_timeout CLIError
    # (rather than hanging the turn forever), which the reply path surfaces like any leg failure.
    release = threading.Event()

    def hang(messages, on_tool=None):
        release.wait(timeout=2.0)  # self-releases so no mutated deadline can wedge the suite
        return "late"

    session, _renderer, _player = make_session(complete_reply=hang)
    try:
        with pytest.raises(CLIError) as excinfo:
            session._complete_within([], timeout=0.05)
        assert excinfo.value.error_type == "agent_timeout"
    finally:
        release.set()  # unblock the abandoned worker so it exits promptly


def test_complete_within_reraises_a_leg_failure_unchanged():
    # A failure the leg raises within the deadline propagates as-is — not masked as a timeout.
    def boom(messages, on_tool=None):
        raise APIError("gateway down")

    session, _renderer, _player = make_session(complete_reply=boom)
    with pytest.raises(APIError, match="gateway down"):
        session._complete_within([], timeout=5.0)


def test_generate_reply_times_out_via_the_backstop(monkeypatch):
    # End-to-end: _generate_reply applies the module deadline, so a stuck thinking leg surfaces
    # an error inline and returns to listening (nothing spoken) instead of hanging the session.
    release = threading.Event()

    def hang(messages, on_tool=None):
        release.wait(timeout=2.0)
        return "late"

    monkeypatch.setattr(engine, "_REPLY_TIMEOUT_SECONDS", 0.05)
    session, renderer, player = make_session(complete_reply=hang)
    try:
        session._generate_reply()
        assert isinstance(session.error, CLIError)
        assert session.error.error_type == "agent_timeout"
        assert any(c[0] == "agent_transcript" and "longer than" in c[1] for c in renderer.calls)
        assert ("reply_done", False) in renderer.calls
        assert player.enqueued == []
    finally:
        release.set()


def test_generate_reply_llm_failure_is_recorded_and_surfaced():
    def boom(messages, on_tool=None):
        del messages
        raise APIError("gateway down")

    session, renderer, player = make_session(complete_reply=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)  # recorded for the exit path
    # Surfaced in the transcript (not swallowed) but nothing is spoken — the turn aborts.
    assert ("agent_transcript", "(error: gateway down)", False) in renderer.calls
    assert ("reply_done", False) in renderer.calls  # the error line is closed off cleanly
    assert player.enqueued == []


def test_generate_reply_tts_failure_midway_is_recorded():
    def boom(text):
        raise APIError("tts down")

    session, renderer, player = make_session(
        complete_reply=lambda m, on_tool=None: "Hi.", synthesize=boom
    )
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert player.enqueued == []
    assert ("reply_started",) in renderer.calls
    assert ("reply_done", False) in renderer.calls


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


# --- helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("end_of_turn", "formatted", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_is_final_turn_with_formatting_waits_for_formatted(end_of_turn, formatted, expected):
    # With formatting on, only a formatted end-of-turn is the cue (better text for the LLM).
    event = _turn("hi", end_of_turn=end_of_turn, turn_is_formatted=formatted)
    assert engine._is_final_turn(event, format_turns=True) is expected


@pytest.mark.parametrize(
    ("end_of_turn", "formatted", "expected"),
    [(True, False, True), (True, True, True), (False, False, False), (False, True, False)],
)
def test_is_final_turn_without_formatting_triggers_on_end_of_turn(end_of_turn, formatted, expected):
    # With --no-format-turns the server never sets turn_is_formatted, so a bare
    # end-of-turn must be the cue — otherwise the agent would never reply.
    event = _turn("hi", end_of_turn=end_of_turn, turn_is_formatted=formatted)
    assert engine._is_final_turn(event, format_turns=False) is expected


def test_is_final_turn_defaults_missing_attrs_to_not_final():
    # A formatted turn missing end_of_turn (and vice versa) must read as not-final, so
    # each absent field defaults to False rather than being treated as present-and-true.
    assert (
        engine._is_final_turn(types.SimpleNamespace(turn_is_formatted=True), format_turns=True)
        is False
    )
    assert (
        engine._is_final_turn(types.SimpleNamespace(end_of_turn=True), format_turns=True) is False
    )


def test_spawn_thread_runs_target():
    ran = threading.Event()
    worker = engine._spawn_thread(ran.set)
    worker.join()
    assert ran.is_set()
    assert worker.is_alive() is False


# --- run_cascade -------------------------------------------------------------


def test_run_cascade_greets_then_pumps_turns():
    def run_stt(on_turn):
        on_turn(_turn("hello"))

    session_box = {}

    def complete_reply(messages, on_tool=None):
        session_box["messages"] = messages
        return "Hi back."

    renderer = FakeRenderer()
    player = FakePlayer()
    config = CascadeConfig(greeting="Welcome.")
    deps = CascadeDeps(
        run_stt=run_stt,
        complete_reply=complete_reply,
        synthesize=lambda text: text.encode(),
        spawn=_sync_spawn,
    )
    run_cascade(renderer=renderer, player=player, config=config, deps=deps)
    assert player.started is True
    assert player.closed is True
    assert ("connected",) in renderer.calls
    assert ("agent_transcript", "Welcome.", False) in renderer.calls
    assert ("user_final", "hello") in renderer.calls
    # The greeting is threaded into the LLM call as prior context.
    assert {"role": "assistant", "content": "Welcome."} in session_box["messages"]


def test_run_cascade_hands_the_session_to_on_session_before_greeting():
    # run_cascade hands the session to on_session before the player starts (TUI wires it).
    captured = {}
    player = FakePlayer()
    deps = CascadeDeps(
        run_stt=lambda on_turn: None,
        complete_reply=lambda m, on_tool=None: "hi",
        synthesize=lambda text: b"",
        spawn=_sync_spawn,
    )
    run_cascade(
        renderer=FakeRenderer(),
        player=player,
        config=CascadeConfig(greeting=""),
        deps=deps,
        on_session=lambda s: captured.update(session=s, started=player.started),
    )
    assert isinstance(captured["session"], CascadeSession)
    assert captured["started"] is False


def test_run_cascade_shuts_down_inflight_worker():
    worker = FakeWorker(alive=True)

    def lazy_spawn(target):
        # Leave the reply "running" so shutdown is what joins it.
        return worker

    def run_stt(on_turn):
        on_turn(_turn("hello"))

    deps = CascadeDeps(
        run_stt=run_stt,
        complete_reply=lambda m, on_tool=None: "hi",
        synthesize=lambda t: b"",
        spawn=lazy_spawn,
    )
    run_cascade(
        renderer=FakeRenderer(), player=FakePlayer(), config=CascadeConfig(greeting=""), deps=deps
    )
    assert worker.joined == 1


def test_run_cascade_reraises_recorded_leg_error():
    def run_stt(on_turn):
        on_turn(_turn("hi"))

    def boom(messages, on_tool=None):
        raise APIError("gateway down")

    deps = CascadeDeps(
        run_stt=run_stt, complete_reply=boom, synthesize=lambda t: b"", spawn=_sync_spawn
    )
    with pytest.raises(APIError, match="gateway down"):
        run_cascade(
            renderer=FakeRenderer(),
            player=FakePlayer(),
            config=CascadeConfig(greeting=""),
            deps=deps,
        )


def test_run_cascade_closes_player_when_stt_raises():
    def run_stt(on_turn):
        raise APIError("stt failed")

    player = FakePlayer()
    deps = CascadeDeps(
        run_stt=run_stt,
        complete_reply=lambda m, on_tool=None: "",
        synthesize=lambda t: b"",
        spawn=_sync_spawn,
    )
    with pytest.raises(APIError, match="stt failed"):
        run_cascade(
            renderer=FakeRenderer(), player=player, config=CascadeConfig(greeting=""), deps=deps
        )
    assert player.closed is True
