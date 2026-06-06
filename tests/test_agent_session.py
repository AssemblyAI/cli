import base64
import json

import pytest

from aai_cli.agent.session import (
    AgentRunConfig,
    VoiceAgentSession,
    _send_audio_loop,
    run_session,
)
from aai_cli.errors import APIError, CLIError, NotAuthenticated


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


class _CloseError(Exception):
    """Mimics websockets.ConnectionClosed carrying a structured close code."""

    def __init__(self, code):
        super().__init__(f"received {code} (policy violation)")
        self.code = code


def test_run_session_connect_auth_failure_raises_not_authenticated():
    def bad_connect(url, **kwargs):
        raise _CloseError(1008)  # Voice Agent rejects a bad key with close 1008

    with pytest.raises(NotAuthenticated):
        run_session(
            "sk_bad",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=[],
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=bad_connect,
        )


def test_run_session_mid_stream_1008_raises_not_authenticated():
    class FakeWS:
        def send(self, _msg):
            pass

        def __iter__(self):
            raise _CloseError(1008)

        def close(self):
            pass

    player = FakePlayer()
    with pytest.raises(NotAuthenticated):
        run_session(
            "sk_bad",
            renderer=FakeRenderer(),
            player=player,
            mic=[],
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=lambda url, **kwargs: FakeWS(),
        )
    assert player.closed is True  # speaker stream still torn down


def test_run_session_surfaces_mic_open_failure_from_capture_thread():
    import threading as _threading

    from aai_cli.errors import CLIError

    class _BoomMic:
        def __iter__(self):
            raise CLIError("no microphone", error_type="mic_error", exit_code=1)

    class _BlockingWS:
        def __init__(self):
            self._closed = _threading.Event()

        def send(self, _msg):
            pass

        def __iter__(self):
            self._closed.wait(timeout=2)  # unblocked when the capture thread closes us
            return iter(())

        def close(self):
            self._closed.set()

    with pytest.raises(CLIError) as exc:
        run_session(
            "sk_live",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=_BoomMic(),
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=lambda url, **kwargs: _BlockingWS(),
        )
    assert exc.value.exit_code == 1  # the real mic failure reaches the user, not a hang


def test_run_session_non_auth_failure_stays_api_error():
    def boom(url, **kwargs):
        raise RuntimeError("network unreachable")

    with pytest.raises(APIError):
        run_session(
            "sk",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=[],
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=boom,
        )


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


def test_run_session_file_driven_stops_after_reply():
    """A file-driven session ends on its own after the agent replies (no hang)."""

    class _ScriptedWS:
        def __init__(self):
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

        def __iter__(self):
            return iter(
                json.dumps(e)
                for e in (
                    {"type": "session.ready"},
                    {"type": "transcript.user", "text": "what time is it"},
                    {"type": "transcript.agent", "text": "it is noon", "interrupted": False},
                    {"type": "reply.done"},
                    # A trailing event the loop must never reach (it should have stopped).
                    {"type": "transcript.user", "text": "SHOULD NOT BE SEEN"},
                )
            )

        def close(self):
            pass

    renderer = FakeRenderer()
    run_session(
        "sk_live",
        renderer=renderer,
        player=FakePlayer(),
        mic=[],  # capture thread waits for ready, then this empty source ends at once
        config=AgentRunConfig(
            voice="ivy",
            system_prompt="x",
            greeting="",
            full_duplex=True,
            exit_after_reply=True,
        ),
        connect=lambda url, **kwargs: _ScriptedWS(),
    )
    finals = [c for c in renderer.calls if c[0] == "user_final"]
    assert ("user_final", "what time is it") in finals
    assert ("user_final", "SHOULD NOT BE SEEN") not in finals  # stopped after the reply


def test_run_session_ws_url_follows_active_environment():
    # The Voice Agent socket must target the active environment's host, not a
    # hardcoded production URL. Capture the URL connect() is handed, short-
    # circuiting with a benign close once we've seen it.
    from aai_cli import environments

    seen: dict[str, str] = {}

    def capture(url, **kwargs):
        seen["url"] = url
        raise _CloseError(1008)

    for env_name, expected in (
        ("sandbox000", "wss://agents.sandbox000.assemblyai-labs.com/v1/ws"),
        ("production", "wss://agents.assemblyai.com/v1/ws"),
    ):
        environments.set_active(environments.get(env_name))
        with pytest.raises(NotAuthenticated):
            run_session(
                "sk",
                renderer=FakeRenderer(),
                player=FakePlayer(),
                mic=[],
                config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
                connect=capture,
            )
        assert seen["url"] == expected
