"""Engine-level tests for the `assembly live` --files feature: the write approver is
threaded into the streaming leg, and a write-approval pause suspends the reply deadline.

Kept in its own module (not appended to the already-large engine suite) and driven against
the shared cascade fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import queue
import types

import pytest

from aai_cli.agent_cascade import brain, engine
from aai_cli.agent_cascade.brain import ApprovalPause, SpeechDelta
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.app.context import AppState
from aai_cli.commands.agent_cascade import _exec
from aai_cli.commands.agent_cascade._exec import run_agent_cascade
from aai_cli.core import config
from tests._cascade_fakes import make_session
from tests.test_agent_cascade_command import _opts


def test_deny_writes_always_rejects():
    # The non-interactive approver declines every write (no channel to confirm one).
    assert _exec._deny_writes("write_file", {"file_path": "/x"}) is False
    assert _exec._deny_writes("edit_file", {"file_path": "/y"}) is False


def test_files_flag_threads_into_config_with_deny_approver_on_headless_path(monkeypatch):
    # --files reaches CascadeConfig.files, and the non-interactive (file source) path wires the
    # deny-writes approver since there's no keyboard channel to confirm a write.
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    captured = {}

    def fake_real(api_key, cfg, *, audio, stt_params, approver=None):
        captured["files"] = cfg.files
        captured["approver"] = approver
        return "deps"

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kwargs: None)
    run_agent_cascade(_opts(source="clip.wav", files=True), AppState(), json_mode=False)
    assert captured["files"] is True
    assert captured["approver"] is _exec._deny_writes


def test_real_passes_approver_to_streamer(monkeypatch):
    # CascadeDeps.real must hand the front-end's write approver to build_streamer so gated
    # writes can be confirmed; on the non-files path it's simply None.
    captured: dict[str, object] = {}

    def fake_build_streamer(api_key, config, *, approver=None):
        captured["approver"] = approver
        return lambda messages: []

    monkeypatch.setattr(engine.brain, "build_streamer", fake_build_streamer)

    def approve(name, args):
        return True

    from assemblyai.streaming.v3 import StreamingParameters

    engine.CascadeDeps.real(
        "k",
        CascadeConfig(files=True),
        audio=iter([]),
        stt_params=StreamingParameters.model_construct(),
        approver=approve,
    )
    assert captured["approver"] is approve


def test_next_event_blocks_with_no_timeout_when_paused():
    # deadline=None means "paused awaiting the user's y/n": block on the queue with no timeout
    # (a slow keypress must never surface a _Timeout), returning the event once it lands.
    session, _renderer, _player = make_session()
    events: queue.Queue = queue.Queue()
    events.put(SpeechDelta("hi"))
    assert session._next_event(events, None, set()) == SpeechDelta("hi")


def test_consume_suspends_then_restores_deadline_across_approval(monkeypatch):
    # An ApprovalPause(active=True) drops the consumer's deadline to None (clock paused); the
    # matching active=False restores a finite deadline — so only the human-think wait is uncounted.
    session, _renderer, _player = make_session()
    events: queue.Queue = queue.Queue()
    for event in (
        ApprovalPause(active=True),
        ApprovalPause(active=False),
        SpeechDelta("Hi."),
        engine._Done(),
    ):
        events.put(event)

    seen: list[float | None] = []
    real_next = session._next_event

    def spy(evts, deadline, before):
        seen.append(deadline)
        return real_next(evts, deadline, before)

    monkeypatch.setattr(session, "_next_event", spy)
    session._consume(events, set(), [])

    assert seen[0] is not None  # initial deadline is finite
    assert seen[1] is None  # paused after ApprovalPause(active=True)
    assert seen[2] is not None  # restored after ApprovalPause(active=False)


def test_approval_deadline_suspends_then_restores_into_the_future():
    # active=True suspends the clock (None); active=False restores a deadline in the FUTURE —
    # asserting it's ahead of now (not merely non-None) pins the + so the timeout actually fires.
    import time

    assert engine._approval_deadline(ApprovalPause(active=True)) is None
    restored = engine._approval_deadline(ApprovalPause(active=False))
    assert restored is not None
    assert restored > time.monotonic()


def test_decide_coerces_non_dict_args_to_empty_dict():
    # When a pending action's args isn't a dict, _decide hands the approver {} (not the raw
    # value). Asserting the approver SAW {} kills the mutant that drops the coercion.
    seen: dict[str, object] = {}

    def approver(name: str, args: dict[str, object]) -> bool:
        seen["name"] = name
        seen["args"] = args
        return True

    decision = brain._decide({"name": "execute", "args": [1, 2]}, approver)

    assert decision == {"type": "approve"}
    assert seen["name"] == "execute"
    assert seen["args"] == {}


def test_decide_passes_dict_args_through_unchanged():
    # When args IS a dict, _decide forwards it verbatim (the `or {}` keeps the real dict). This
    # kills the Or->And mutant, which would collapse a real dict to {} before the approver sees it.
    seen: dict[str, object] = {}

    def approver(name: str, args: dict[str, object]) -> bool:
        seen["args"] = args
        return True

    brain._decide({"name": "write_file", "args": {"file_path": "n.txt"}}, approver)

    assert seen["args"] == {"file_path": "n.txt"}


def test_brain_stream_event_dataclasses_are_frozen():
    # SpeechDelta/ToolNotice/ApprovalPause are frozen; the frozen=True->False mutant is killed
    # by asserting a write raises. A variable attr name dodges ruff B010 and pyright's frozen check.
    import dataclasses

    probe = "injected_probe"
    events = (
        brain.SpeechDelta(text="x"),
        brain.ToolNotice(label="Searching", fillers=("one moment",)),
        brain.ApprovalPause(active=True),
    )
    for event in events:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(event, probe, 1)


class _SpyGatedGraph:
    """A graph satisfying _GatedGraph that records get_state calls (the gate inspection)."""

    def __init__(self) -> None:
        self.get_state_calls = 0

    def invoke(self, input, config=None):  # satisfies CompiledAgent (unused by the stream path)
        return {}

    def stream(self, graph_input, config, *, stream_mode):
        return iter(())  # no chunks; the test only cares which path runs

    def get_state(self, config):
        self.get_state_calls += 1
        return types.SimpleNamespace(interrupts=())


def test_stream_graph_defaults_to_ungated():
    # _stream_graph's `gated` defaults to False: an ungated pass never inspects interrupts. The
    # gated=False->True mutant would route a _GatedGraph through _stream_gated -> get_state, so
    # asserting get_state is never called kills it.
    graph = _SpyGatedGraph()

    list(brain._stream_graph(graph, []))

    assert graph.get_state_calls == 0
