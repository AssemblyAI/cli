import base64
import json

import pytest

from assemblyai_cli.agent.session import VoiceAgentSession, _send_audio_loop, run_session
from assemblyai_cli.errors import APIError, CLIError, NotAuthenticated


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


def _session(*, full_duplex=False):
    return VoiceAgentSession(
        renderer=FakeRenderer(),
        player=FakePlayer(),
        full_duplex=full_duplex,
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
    assert ("user_partial", "what") in s.renderer.calls
    assert ("user_final", "what time") in s.renderer.calls
    assert ("agent_transcript", "noon", False) in s.renderer.calls


def test_unauthorized_error_raises_cli_error_exit_2():
    s = _session()
    with pytest.raises(CLIError) as excinfo:
        s.dispatch({"type": "session.error", "code": "UNAUTHORIZED", "message": "bad key"})
    assert excinfo.value.exit_code == 2


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


_POLICY_VIOLATION = "received 1008 (policy violation); then sent 1008 (policy violation)"


def test_run_session_connect_auth_failure_raises_not_authenticated():
    def bad_connect(url, **kwargs):
        raise RuntimeError(_POLICY_VIOLATION)

    with pytest.raises(NotAuthenticated):
        run_session(
            "sk_bad",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=[],
            voice="ivy",
            system_prompt="x",
            greeting="hi",
            connect=bad_connect,
        )


def test_run_session_mid_stream_1008_raises_not_authenticated():
    class FakeWS:
        def send(self, _msg):
            pass

        def __iter__(self):
            raise RuntimeError(_POLICY_VIOLATION)

        def close(self):
            pass

    player = FakePlayer()
    with pytest.raises(NotAuthenticated):
        run_session(
            "sk_bad",
            renderer=FakeRenderer(),
            player=player,
            mic=[],
            voice="ivy",
            system_prompt="x",
            greeting="hi",
            connect=lambda url, **kwargs: FakeWS(),
        )
    assert player.closed is True  # speaker stream still torn down


def test_run_session_non_auth_failure_stays_api_error():
    def boom(url, **kwargs):
        raise RuntimeError("network unreachable")

    with pytest.raises(APIError):
        run_session(
            "sk",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=[],
            voice="ivy",
            system_prompt="x",
            greeting="hi",
            connect=boom,
        )


def test_full_duplex_reply_started_announces_without_muting():
    s = _session(full_duplex=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is False
    assert ("reply_started",) in s.renderer.calls
