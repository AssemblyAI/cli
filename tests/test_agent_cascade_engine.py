"""Orchestration tests for the terminal voice cascade (aai_cli.agent_cascade.engine).

The cascade's three network legs and its thread spawner are injected through
CascadeDeps, so every test here runs against fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import threading
import types

import pytest

from aai_cli.agent_cascade import _runtime, engine
from aai_cli.agent_cascade.brain import SpeechDelta, ToolNotice
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.engine import CascadeDeps, CascadeSession, run_cascade
from aai_cli.agent_cascade.plan import TodoItem, TodoUpdate
from aai_cli.core.errors import APIError
from tests._cascade_fakes import FakePlayer, FakeRenderer, FakeWorker, make_session
from tests._cascade_fakes import deltas as _deltas
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
    def boom(text, sink):
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
    session, renderer, player = make_session(stream_reply=_deltas("Sure thing."))
    session.on_turn(_turn("what time is it"))
    assert ("user_final", "what time is it") in renderer.calls
    assert {"role": "user", "content": "what time is it"} in session.history
    assert {"role": "assistant", "content": "Sure thing."} in session.history
    assert player.enqueued == [b"pcm:Sure thing."]
    assert ("reply_done", False) in renderer.calls


def test_empty_reply_still_brackets_reply_done_with_reply_started():
    # A reply that produces no clause and no tool call must still emit reply_started before
    # reply_done, so a stream consumer never sees an unmatched reply_done and the TUI resets.
    session, renderer, _player = make_session(stream_reply=lambda m: [])
    session._generate_reply()
    assert renderer.calls.count(("reply_started",)) == 1
    started = renderer.calls.index(("reply_started",))
    done = renderer.calls.index(("reply_done", False))
    assert started < done


def test_reply_started_fires_exactly_once_per_turn():
    # The idempotent _emit_reply_started must not double-fire: a multi-clause spoken reply emits
    # reply_started once, and the per-turn reset means a second turn emits it again (once).
    session, renderer, _player = make_session(stream_reply=_deltas("One. ", "Two. "))
    session._generate_reply()
    session._generate_reply()
    assert renderer.calls.count(("reply_started",)) == 2


def test_reply_forwards_tool_calls_to_the_renderer():
    def stream(messages):
        yield ToolNotice("Searching the web", ("Searching now.",))
        yield SpeechDelta("Found it.")

    session, renderer, _player = make_session(stream_reply=stream)
    session.on_turn(_turn("what's the news"))
    assert ("tool_call", "Searching the web") in renderer.calls


def test_reply_forwards_a_plan_and_still_speaks_the_reply():
    # A TodoUpdate is a purely visual affordance: it reaches the renderer as todos_updated and,
    # unlike a ToolNotice, does NOT suppress the spoken reply that follows it.
    todos = (TodoItem(content="Book a flight", status="in_progress"),)

    def stream(messages):
        yield TodoUpdate(todos)
        yield SpeechDelta("First I'll book the flight.")

    session, renderer, _player = make_session(stream_reply=stream)
    session.on_turn(_turn("plan my trip"))
    assert ("todos_updated", todos) in renderer.calls
    # The reply was still spoken (a plain ToolNotice would have held it back).
    assert ("agent_transcript", "First I'll book the flight.", False) in renderer.calls


def test_on_turn_interim_shows_partial_and_does_not_reply():
    streamed = []
    session, renderer, _player = make_session(
        stream_reply=lambda m: streamed.append(m) or [SpeechDelta("x")]
    )
    session.on_turn(_turn("partial words", end_of_turn=False))
    assert ("user_partial", "partial words") in renderer.calls
    assert streamed == []  # no reply generated for an interim turn
    assert session.history == []


def test_on_turn_interim_barges_in_on_live_reply():
    session, _renderer, player = make_session()
    session._reply = FakeWorker(alive=True)
    session.on_turn(_turn("uh", end_of_turn=False))
    assert player.flushed == 1
    assert session._reply is None


# --- spoken approval (--files): route the next final transcript to the open gate -------------


def test_on_turn_routes_final_to_voice_during_approval_pause():
    # While a --files write/run awaits approval, the next FINAL transcript answers the gate by
    # voice — it does NOT render a user turn or start a new reply.
    session, renderer, _player = make_session()
    voiced: list[str] = []
    session.on_approval_voice = voiced.append
    session._set_awaiting_approval(active=True)

    session.on_turn(_turn("yes, run it"))

    assert voiced == ["yes, run it"]
    assert session.history == []  # no new turn started
    assert ("user_final", "yes, run it") not in renderer.calls


def test_on_turn_ignores_interim_during_approval_pause():
    # Interim partials during the pause are dropped (only a final transcript answers the gate).
    session, renderer, _player = make_session()
    voiced: list[str] = []
    session.on_approval_voice = voiced.append
    session._set_awaiting_approval(active=True)

    session.on_turn(_turn("yes", end_of_turn=False))

    assert voiced == []
    assert renderer.calls == []


def test_on_turn_final_during_pause_without_voice_sink_is_dropped():
    # Keyboard-only path (no voice sink): a final transcript during the pause is simply dropped,
    # not started as a turn — pins the `on_approval_voice is not None` guard.
    session, renderer, _player = make_session()  # on_approval_voice defaults to None
    session._set_awaiting_approval(active=True)

    session.on_turn(_turn("anything"))

    assert session.history == []
    assert renderer.calls == []


def test_on_turn_resumes_normal_turns_once_approval_clears():
    # After the pause clears (active=False), a final transcript starts a reply again, NOT voice.
    session, renderer, _player = make_session(stream_reply=_deltas("Done."))
    voiced: list[str] = []
    session.on_approval_voice = voiced.append
    session._set_awaiting_approval(active=True)
    session._set_awaiting_approval(active=False)

    session.on_turn(_turn("what time is it"))

    assert voiced == []
    assert ("user_final", "what time is it") in renderer.calls


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
    worker = _runtime.spawn_thread(ran.set)
    worker.join()
    assert ran.is_set()
    assert worker.is_alive() is False


# --- run_cascade -------------------------------------------------------------


def test_run_cascade_greets_then_pumps_turns():
    def run_stt(on_turn):
        on_turn(_turn("hello"))

    session_box = {}

    def stream_reply(messages):
        session_box["messages"] = messages
        return [SpeechDelta("Hi back.")]

    renderer = FakeRenderer()
    player = FakePlayer()
    config = CascadeConfig(greeting="Welcome.")
    deps = CascadeDeps(
        run_stt=run_stt,
        stream_reply=stream_reply,
        synthesize=lambda text, sink: sink(text.encode()),
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
        stream_reply=_deltas("hi"),
        synthesize=lambda text, sink: sink(b""),
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
        stream_reply=_deltas("hi"),
        synthesize=lambda text, sink: sink(b""),
        spawn=lazy_spawn,
    )
    run_cascade(
        renderer=FakeRenderer(), player=FakePlayer(), config=CascadeConfig(greeting=""), deps=deps
    )
    assert worker.joined == 1


def test_run_cascade_reraises_recorded_leg_error():
    def run_stt(on_turn):
        on_turn(_turn("hi"))

    def boom(messages):
        raise APIError("gateway down")

    deps = CascadeDeps(
        run_stt=run_stt,
        stream_reply=boom,
        synthesize=lambda text, sink: sink(b""),
        spawn=_sync_spawn,
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
        stream_reply=_deltas(""),
        synthesize=lambda text, sink: sink(b""),
        spawn=_sync_spawn,
    )
    with pytest.raises(APIError, match="stt failed"):
        run_cascade(
            renderer=FakeRenderer(), player=player, config=CascadeConfig(greeting=""), deps=deps
        )
    assert player.closed is True


def test_runtime_reply_sentinels_are_frozen():
    # Done/Failure/Timeout are frozen dataclasses; the frozen=True->False mutant is killed by
    # asserting a write raises. A variable attr name dodges ruff B010 and pyright's frozen check.
    import dataclasses

    probe = "injected_probe"
    for instance in (_runtime.Done(), _runtime.Failure(error=APIError("x")), _runtime.Timeout()):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(instance, probe, 1)


def test_cascade_session_speaking_event_is_not_an_init_field():
    # _speaking is internal state, never a constructor argument; the init=False->True mutant is
    # killed by asserting the field stays init=False.
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(engine.CascadeSession)}
    assert fields["_speaking"].init is False


def test_detach_executor_threads_noop_without_registry(monkeypatch):
    # When concurrent.futures exposes no thread registry, detach returns before touching it. A
    # thread that WOULD be popped is staged, so the mutant dropping the early return crashes on
    # None.pop and the test kills it; with the return intact the call is a clean no-op.
    monkeypatch.setattr(_runtime.cf_thread, "_threads_queues", None, raising=False)
    staged = threading.Thread(target=lambda: None)
    monkeypatch.setattr(_runtime, "executor_threads", lambda: {staged})

    _runtime.detach_executor_threads_since(set())  # no AttributeError: early-returns on None
