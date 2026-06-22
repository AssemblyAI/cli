"""Engine-level tests for the `assembly live` --files feature: the write approver is
threaded into the streaming leg, and a write-approval pause suspends the reply deadline.

Kept in its own module (not appended to the already-large engine suite) and driven against
the shared cascade fakes — no sockets, mic, or speaker.
"""

from __future__ import annotations

import queue
import types

from aai_cli.agent_cascade import engine
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
