"""Engine-level tests for the `assembly live` --files feature: the write approver is
threaded into the streaming leg, and a write-approval pause suspends the reply deadline.

Kept in its own module (not appended to the already-large engine suite) and driven against
the shared cascade fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import queue

from aai_cli.agent_cascade import engine
from aai_cli.agent_cascade.brain import ApprovalPause, SpeechDelta
from aai_cli.agent_cascade.config import CascadeConfig
from tests._cascade_fakes import make_session


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
