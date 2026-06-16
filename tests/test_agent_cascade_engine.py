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
from aai_cli.core.errors import APIError


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def connected(self):
        self.calls.append(("connected",))

    def user_partial(self, text):
        self.calls.append(("user_partial", text))

    def user_final(self, text):
        self.calls.append(("user_final", text))

    def reply_started(self):
        self.calls.append(("reply_started",))

    def agent_transcript(self, text, *, interrupted):
        self.calls.append(("agent_transcript", text, interrupted))

    def reply_done(self, *, interrupted):
        self.calls.append(("reply_done", interrupted))


class FakePlayer:
    def __init__(self):
        self.enqueued = []
        self.flushed = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def enqueue(self, pcm):
        self.enqueued.append(pcm)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


class FakeWorker:
    def __init__(self, *, alive):
        self._alive = alive
        self.joined = 0

    def is_alive(self):
        return self._alive

    def join(self):
        self.joined += 1
        self._alive = False


def _sync_spawn(target):
    """Run the reply body inline and hand back a finished worker, so the cascade is
    driven deterministically without real threads."""
    target()
    return FakeWorker(alive=False)


def _turn(text, *, end_of_turn=True, turn_is_formatted=True):
    return types.SimpleNamespace(
        transcript=text, end_of_turn=end_of_turn, turn_is_formatted=turn_is_formatted
    )


def make_session(
    *,
    complete_reply=lambda messages: "Hello there.",
    synthesize=lambda text: b"pcm:" + text.encode(),
    spawn=_sync_spawn,
    run_stt=lambda on_turn: None,
    config=None,
):
    deps = CascadeDeps(
        run_stt=run_stt, complete_reply=complete_reply, synthesize=synthesize, spawn=spawn
    )
    renderer = FakeRenderer()
    player = FakePlayer()
    session = CascadeSession(
        deps=deps, renderer=renderer, player=player, config=config or CascadeConfig()
    )
    return session, renderer, player


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
    session, renderer, player = make_session(complete_reply=lambda m: "Sure thing.")
    session.on_turn(_turn("what time is it"))
    assert ("user_final", "what time is it") in renderer.calls
    assert {"role": "user", "content": "what time is it"} in session.history
    assert {"role": "assistant", "content": "Sure thing."} in session.history
    assert player.enqueued == [b"pcm:Sure thing."]
    assert ("reply_done", False) in renderer.calls


def test_on_turn_interim_shows_partial_and_does_not_reply():
    replies = []
    session, renderer, _player = make_session(complete_reply=lambda m: replies.append(m) or "x")
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
        complete_reply=lambda m: "One. Two! Three?",
        synthesize=lambda text: spoken.append(text) or text.encode(),
    )
    session._generate_reply()
    assert spoken == ["One.", "Two!", "Three?"]
    assert player.enqueued == [b"One.", b"Two!", b"Three?"]
    assert ("reply_started",) in renderer.calls
    assert ("agent_transcript", "One.", False) in renderer.calls
    assert session.history[-1] == {"role": "assistant", "content": "One. Two! Three?"}
    assert ("reply_done", False) in renderer.calls


def test_generate_reply_threads_system_prompt_and_history():
    captured = {}

    def capture(messages):
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
        complete_reply=lambda m: "a. b.", config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "user", "content": "hi"})
    session._generate_reply()
    # user + assistant would be 2; the window caps it to the most recent 1.
    assert session.history == [{"role": "assistant", "content": "a. b."}]


def test_on_turn_trims_history_window():
    # An empty reply adds no assistant turn, so only on_turn's own trim caps the list.
    session, _renderer, _player = make_session(
        complete_reply=lambda m: "", config=CascadeConfig(max_history=1)
    )
    session.history.append({"role": "assistant", "content": "old"})
    session.on_turn(_turn("newest"))
    assert session.history == [{"role": "user", "content": "newest"}]


def test_generate_reply_stop_after_first_sentence_records_partial():
    def synth(text):
        if text == "Two.":
            session._stop.set()
        return text.encode()

    session, renderer, player = make_session(complete_reply=lambda m: "One. Two. Three.")
    session.deps.synthesize = synth
    session._generate_reply()
    # Only the first sentence finished enqueuing before the barge-in stop landed.
    assert player.enqueued == [b"One."]
    assert session.history[-1] == {"role": "assistant", "content": "One."}
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_stop_before_first_sentence_speaks_nothing():
    session, renderer, player = make_session(complete_reply=lambda m: "One. Two.")
    session._stop.set()
    session._generate_reply()
    assert player.enqueued == []
    # nothing spoken -> no assistant turn recorded
    assert all(item.get("role") != "assistant" for item in session.history)
    assert ("reply_done", True) in renderer.calls


def test_generate_reply_llm_failure_is_recorded_and_aborts():
    def boom(messages):
        raise APIError("gateway down")

    session, renderer, _player = make_session(complete_reply=boom)
    session._generate_reply()
    assert isinstance(session.error, APIError)
    assert ("reply_started",) not in renderer.calls  # aborted before speaking


def test_generate_reply_tts_failure_midway_is_recorded():
    def boom(text):
        raise APIError("tts down")

    session, renderer, player = make_session(complete_reply=lambda m: "Hi.", synthesize=boom)
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
    session, _renderer, player = make_session()
    worker = FakeWorker(alive=True)
    session._reply = worker
    session._barge_in()
    assert session._stop.is_set()
    assert player.flushed == 1
    assert worker.joined == 1
    assert session._reply is None


def test_barge_in_no_worker_does_not_flush():
    session, _renderer, player = make_session()
    session._barge_in()
    assert player.flushed == 0


def test_barge_in_finished_worker_does_not_flush():
    session, _renderer, player = make_session()
    session._reply = FakeWorker(alive=False)
    session._barge_in()
    assert player.flushed == 0
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

    def complete_reply(messages):
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


def test_run_cascade_shuts_down_inflight_worker():
    worker = FakeWorker(alive=True)

    def lazy_spawn(target):
        # Leave the reply "running" so shutdown is what joins it.
        return worker

    def run_stt(on_turn):
        on_turn(_turn("hello"))

    deps = CascadeDeps(
        run_stt=run_stt, complete_reply=lambda m: "hi", synthesize=lambda t: b"", spawn=lazy_spawn
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
        run_stt=run_stt, complete_reply=lambda m: "", synthesize=lambda t: b"", spawn=_sync_spawn
    )
    with pytest.raises(APIError, match="stt failed"):
        run_cascade(
            renderer=FakeRenderer(), player=player, config=CascadeConfig(greeting=""), deps=deps
        )
    assert player.closed is True
