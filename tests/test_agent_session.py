"""VoiceAgentSession dispatch/gate-level tests.

run_session-level tests (connection lifecycle, error classification) live in
test_agent_session_run.py.
"""

import base64
import json

import pytest

from aai_cli.agent.session import VoiceAgentSession, _send_audio_loop
from aai_cli.errors import APIError, NotAuthenticated


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

    def enqueue(self, pcm):
        self.enqueued.append(pcm)

    def flush(self):
        self.flushed += 1

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


def _session(*, full_duplex=False, exit_after_reply=False, ready_event=None):
    return VoiceAgentSession(
        renderer=FakeRenderer(),
        player=FakePlayer(),
        full_duplex=full_duplex,
        exit_after_reply=exit_after_reply,
        ready_event=ready_event,
    )


def test_ready_opens_gate_and_announces():
    s = _session()
    assert s.ready is False
    s.dispatch({"type": "session.ready", "session_id": "sess_1"})
    assert s.ready is True
    assert ("connected",) in s.renderer.calls


def test_half_duplex_mutes_during_reply():
    s = _session(full_duplex=False)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is True
    s.dispatch({"type": "reply.done"})
    assert s.muted is False


def test_full_duplex_never_mutes_and_flushes_on_speech_start():
    s = _session(full_duplex=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is False
    s.dispatch({"type": "input.speech.started"})
    assert s.player.flushed == 1


def test_reply_audio_is_decoded_and_enqueued():
    import base64

    s = _session()
    payload = base64.b64encode(b"\x10\x20").decode()
    s.dispatch({"type": "reply.audio", "data": payload})
    assert s.player.enqueued == [b"\x10\x20"]


def test_interrupted_reply_done_flushes_playback():
    s = _session()
    s.dispatch({"type": "reply.done", "status": "interrupted"})
    assert s.player.flushed == 1
    assert ("reply_done", True) in s.renderer.calls


def test_transcripts_routed_to_renderer():
    s = _session()
    s.dispatch({"type": "transcript.user.delta", "text": "what"})
    s.dispatch({"type": "transcript.user", "text": "what time"})
    s.dispatch({"type": "transcript.agent", "text": "noon", "interrupted": False})
    # An agent transcript with no "interrupted" key defaults to False (pins the default).
    s.dispatch({"type": "transcript.agent", "text": "later"})
    assert ("user_partial", "what") in s.renderer.calls
    assert ("user_final", "what time") in s.renderer.calls
    assert ("agent_transcript", "noon", False) in s.renderer.calls
    assert ("agent_transcript", "later", False) in s.renderer.calls


def test_unauthorized_error_raises_not_authenticated_exit_4():
    # A rejected voice-agent connection uses the same exit 4 as every other
    # not-authenticated path across the CLI (REST, streaming), not a one-off code.
    s = _session()
    with pytest.raises(NotAuthenticated) as excinfo:
        s.dispatch({"type": "session.error", "code": "UNAUTHORIZED", "message": "bad key"})
    assert excinfo.value.exit_code == 4
    assert "bad key" in str(excinfo.value)  # the server message wins over code/fallback
    assert excinfo.value.suggestion is not None and "assembly login" in excinfo.value.suggestion
    # A presented-and-refused key: auto-login must not retry over a bad env key.
    assert excinfo.value.rejected_key is True


def test_other_session_error_raises_api_error():
    s = _session()
    with pytest.raises(APIError):
        s.dispatch({"type": "session.error", "code": "invalid_value", "message": "bad voice"})


def test_unknown_and_tool_events_are_ignored():
    s = _session()
    s.dispatch({"type": "tool.call", "call_id": "c1", "name": "x", "arguments": {}})
    s.dispatch({"type": "something.new"})
    assert s.renderer.calls == []  # nothing surfaced, no exception


def test_reply_audio_without_data_is_ignored():
    s = _session()
    s.dispatch({"type": "reply.audio"})  # no data key
    s.dispatch({"type": "reply.audio", "data": ""})  # empty data
    assert s.player.enqueued == []


def test_should_send_audio_only_when_ready_and_unmuted():
    s = _session()
    assert s.should_send_audio() is False  # not ready yet
    s.ready = True
    assert s.should_send_audio() is True
    s.muted = True
    assert s.should_send_audio() is False  # gated while the agent speaks


def test_should_send_audio_reflects_dispatch_mute_cycle():
    # The duplex gate is driven entirely by dispatched events (the path the receive
    # loop uses); should_send_audio reads it through the lock from the capture thread.
    s = _session(full_duplex=False)
    s.dispatch({"type": "session.ready"})
    assert s.should_send_audio() is True
    s.dispatch({"type": "reply.started"})
    assert s.should_send_audio() is False  # muted while the agent speaks
    s.dispatch({"type": "reply.done"})
    assert s.should_send_audio() is True  # gate reopens once the reply finishes


class _RecordingWS:
    def __init__(self, fail_on_send=False):
        self.sent = []
        self.fail_on_send = fail_on_send

    def send(self, msg):
        if self.fail_on_send:
            raise RuntimeError("socket closed")
        self.sent.append(msg)


def test_send_audio_loop_forwards_frames_when_gate_open():
    s = _session()
    s.ready = True  # gate open, not muted
    ws = _RecordingWS()
    _send_audio_loop(ws, s, [b"\x01\x02", b"\x03\x04"])
    payloads = [json.loads(m) for m in ws.sent]
    assert [p["type"] for p in payloads] == ["input.audio", "input.audio"]
    assert base64.b64decode(payloads[0]["audio"]) == b"\x01\x02"


def test_send_audio_loop_drops_frames_while_muted():
    s = _session()
    s.ready = True
    s.muted = True  # gate closed -> frames dropped, nothing sent
    ws = _RecordingWS()
    _send_audio_loop(ws, s, [b"\x01\x02"])
    assert ws.sent == []


def test_send_audio_loop_stops_on_send_error():
    s = _session()
    s.ready = True
    ws = _RecordingWS(fail_on_send=True)
    # Must return (not raise) when the socket is gone mid-send.
    _send_audio_loop(ws, s, [b"\x01\x02", b"\x03\x04"])


def test_full_duplex_reply_started_announces_without_muting():
    s = _session(full_duplex=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is False
    assert ("reply_started",) in s.renderer.calls


def test_ready_sets_ready_event():
    import threading

    ev = threading.Event()
    s = _session(exit_after_reply=True, ready_event=ev)
    assert ev.is_set() is False
    s.dispatch({"type": "session.ready"})
    assert ev.is_set() is True  # capture thread is now free to stream the file


def test_exit_after_reply_finishes_after_user_then_reply_done():
    s = _session(full_duplex=True, exit_after_reply=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "transcript.user", "text": "hello there"})
    assert s.finished is False  # not until the agent has actually replied
    s.dispatch({"type": "reply.done"})
    assert s.finished is True


def test_exit_after_reply_ignores_greeting_reply_before_user_speech():
    s = _session(full_duplex=True, exit_after_reply=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.done"})  # e.g. a greeting, before any user speech
    assert s.finished is False


def test_exit_after_reply_ignores_interrupted_reply():
    s = _session(full_duplex=True, exit_after_reply=True)
    s.dispatch({"type": "transcript.user", "text": "hi"})
    s.dispatch({"type": "reply.done", "status": "interrupted"})
    assert s.finished is False


def test_exit_after_reply_off_never_finishes():
    s = _session(full_duplex=True, exit_after_reply=False)
    s.dispatch({"type": "transcript.user", "text": "hi"})
    s.dispatch({"type": "reply.done"})
    assert s.finished is False  # live mic sessions run until Ctrl-C


def test_send_audio_loop_waits_for_ready_event_before_streaming():
    import threading

    ev = threading.Event()
    ev.set()  # already ready -> loop proceeds immediately
    s = _session(exit_after_reply=True, ready_event=ev)
    s.ready = True
    ws = _RecordingWS()
    _send_audio_loop(ws, s, [b"\x01\x02"])
    assert len(ws.sent) == 1  # frame forwarded once the gate is open


def test_send_audio_loop_waits_on_ready_event_with_bounded_timeout():
    # The wait on ready_event is bounded so a server that never sends `ready` can't
    # wedge the send loop forever; pins the 10s timeout.
    seen = {}

    class _RecordingEvent:
        def wait(self, timeout=None):
            seen["timeout"] = timeout
            return True

    s = _session(exit_after_reply=True, ready_event=_RecordingEvent())
    s.ready = True
    _send_audio_loop(_RecordingWS(), s, [b"\x01\x02"])
    assert seen["timeout"] == 10


def test_send_audio_loop_fails_loudly_when_ready_times_out():
    # A server that never sends session.ready must fail the run, not silently
    # consume (and drop) the finite file source frame by frame.
    class _NeverReadyEvent:
        def wait(self, timeout=None):
            return False

    s = _session(exit_after_reply=True, ready_event=_NeverReadyEvent())
    s.ready = True  # even an open gate must not be reached past a timed-out wait
    ws = _RecordingWS()
    with pytest.raises(APIError) as exc:
        _send_audio_loop(ws, s, [b"\x01\x02"])
    assert "ready" in exc.value.message
    assert ws.sent == []  # no audio was consumed or sent
