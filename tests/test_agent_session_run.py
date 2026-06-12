"""run_session-level tests: connection lifecycle, error classification, env URLs.

Dispatch/gate-level VoiceAgentSession tests live in test_agent_session.py.
"""

import json
import logging
import types

import pytest

from aai_cli.agent.session import AgentRunConfig, run_session
from aai_cli.errors import APIError, CLIError, NotAuthenticated
from aai_cli.ws import WEBSOCKETS_LOGGERS


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


class _RecordingWS:
    def __init__(self, fail_on_send=False):
        self.sent = []
        self.fail_on_send = fail_on_send

    def send(self, msg):
        if self.fail_on_send:
            raise RuntimeError("socket closed")
        self.sent.append(msg)


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


def test_run_session_capture_thread_is_daemon(monkeypatch):
    # The capture thread is a daemon so a stuck mic read can't keep the process alive
    # after the session ends.
    import threading as _threading

    from aai_cli.agent import session as session_mod

    daemons = []
    real_cls = session_mod.threading.Thread

    class SpyThread(real_cls):
        def __init__(self, *a, **k):
            daemons.append(k.get("daemon"))
            super().__init__(*a, **k)

    monkeypatch.setattr(session_mod.threading, "Thread", SpyThread)

    class _BoomMic:
        def __iter__(self):
            raise CLIError("no microphone", error_type="mic_error", exit_code=1)

    class _BlockingWS:
        def __init__(self):
            self._closed = _threading.Event()

        def send(self, _msg):
            pass

        def __iter__(self):
            self._closed.wait(timeout=2)
            return iter(())

        def close(self):
            self._closed.set()

    with pytest.raises(CLIError):
        run_session(
            "sk_live",
            renderer=FakeRenderer(),
            player=FakePlayer(),
            mic=_BoomMic(),
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=lambda url, **kwargs: _BlockingWS(),
        )
    assert daemons == [True]  # the one capture thread, created as a daemon


def test_run_session_does_not_close_player_that_failed_to_open():
    # If opening the speaker stream raises, the cleanup must NOT call close() on a
    # player that never started (pins the player_started=False initializer).
    class _FailingPlayer(FakePlayer):
        def start(self):
            raise CLIError("speaker busy", error_type="audio_output_error", exit_code=1)

    player = _FailingPlayer()
    with pytest.raises(CLIError):
        run_session(
            "sk",
            renderer=FakeRenderer(),
            player=player,
            mic=[],
            config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
            connect=lambda url, **kwargs: _RecordingWS(),
        )
    assert player.closed is False  # never opened, so never closed


class _HandshakeRejected(Exception):
    """Mimics websockets' InvalidStatus: a structured HTTP status on ``.response``."""

    def __init__(self, status):
        super().__init__(f"server rejected WebSocket connection: HTTP {status}")
        self.response = types.SimpleNamespace(status_code=status)


def _run_with_connect(connect):
    run_session(
        "sk",
        renderer=FakeRenderer(),
        player=FakePlayer(),
        mic=[],
        config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
        connect=connect,
    )


def test_run_session_handshake_403_is_api_error_like_stream():
    # Harmonized with `stream`: a plain handshake 403 is an API error (exit 1), not
    # "Your API key was rejected" — 403 also covers non-credential blocks.
    def reject(url, **kwargs):
        raise _HandshakeRejected(403)

    with pytest.raises(APIError) as exc:
        _run_with_connect(reject)
    assert exc.value.error_type == "api_error"
    assert exc.value.exit_code == 1
    assert "Could not connect to the voice agent" in exc.value.message
    assert "HTTP 403" in exc.value.message
    # The rejected handshake carries the actionable next steps, env host included.
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion
    assert "access to agents.assemblyai.com." in exc.value.suggestion


def test_run_session_handshake_401_is_still_auth_failure():
    # A genuinely auth-shaped rejection (HTTP 401) keeps the rejected-key path.
    def reject(url, **kwargs):
        raise _HandshakeRejected(401)

    with pytest.raises(NotAuthenticated) as exc:
        _run_with_connect(reject)
    assert exc.value.exit_code == 4
    assert exc.value.rejected_key is True
    assert exc.value.suggestion is not None
    assert "assembly whoami" in exc.value.suggestion


def test_run_session_auth_worded_failure_is_still_auth_failure():
    # The text heuristic ("unauthorized" etc.) keeps working for real bad keys.
    def reject(url, **kwargs):
        raise RuntimeError("connection rejected: Unauthorized")

    with pytest.raises(NotAuthenticated):
        _run_with_connect(reject)


class _CleanWS:
    def send(self, _msg):
        pass

    def __iter__(self):
        return iter(())

    def close(self):
        pass


def test_run_session_silences_websockets_loggers():
    # websockets' sync reader thread logs teardown errors (EOFError tracebacks) via
    # its own loggers; run_session must mute them so they never hit the user's stderr.
    loggers = [logging.getLogger(name) for name in WEBSOCKETS_LOGGERS]
    previous = [lg.level for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.NOTSET)
        _run_with_connect(lambda url, **kwargs: _CleanWS())
        for lg in loggers:
            assert lg.level == logging.CRITICAL
            assert not lg.isEnabledFor(logging.ERROR)  # an ERROR record is dropped
    finally:
        for lg, level in zip(loggers, previous, strict=True):
            lg.setLevel(level)


def test_websockets_logger_names_cover_the_sync_client():
    # The sync client logs through "websockets.client"; pin that the silenced set
    # covers it (and the parent, for any future child loggers).
    assert "websockets.client" in WEBSOCKETS_LOGGERS
    assert "websockets" in WEBSOCKETS_LOGGERS


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


def test_run_session_ws_url_follows_active_environment() -> None:
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


def test_run_session_defaults_to_websockets_sync_connect(monkeypatch):
    # With no injected connect, run_session lazily imports websockets' sync client
    # (pins the `connect is None` default-import branch). Patch the import target so
    # no real socket is opened; an empty message stream ends the loop immediately.
    class _DefaultCleanWS:
        def send(self, _msg):
            pass

        def __iter__(self):
            return iter(())

        def close(self):
            pass

    monkeypatch.setattr("websockets.sync.client.connect", lambda url, **kwargs: _DefaultCleanWS())
    run_session(
        "sk_live",
        renderer=FakeRenderer(),
        player=FakePlayer(),
        mic=[],
        config=AgentRunConfig(voice="ivy", system_prompt="x", greeting="hi"),
    )
