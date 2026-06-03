import pytest

from assemblyai_cli.agent.session import VoiceAgentSession
from assemblyai_cli.errors import APIError, CLIError


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

    def enqueue(self, pcm):
        self.enqueued.append(pcm)

    def flush(self):
        self.flushed += 1


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


def test_full_duplex_reply_started_announces_without_muting():
    s = _session(full_duplex=True)
    s.dispatch({"type": "session.ready"})
    s.dispatch({"type": "reply.started"})
    assert s.muted is False
    assert ("reply_started",) in s.renderer.calls
