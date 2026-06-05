from __future__ import annotations

import base64
import contextlib
import json
import threading
from collections.abc import Callable
from typing import Any

from aai_cli import environments
from aai_cli.errors import APIError, CLIError, auth_failure, is_auth_failure


def _ws_url() -> str:
    """Voice Agent socket URL for the active environment (set at CLI startup)."""
    return f"wss://{environments.active().agents_host}/v1/ws"


DEFAULT_PROMPT = (
    "You are a friendly voice assistant having a casual conversation. Keep replies "
    "short and natural, usually one or two sentences. Speak the way a person would "
    "in real conversation: relaxed, low-key, no exclamation marks."
)
DEFAULT_GREETING = "Hey, what's on your mind?"

# session.error codes that mean the connection is unauthorized -> exit 2.
_AUTH_ERROR_CODES = {"UNAUTHORIZED", "FORBIDDEN"}


class VoiceAgentSession:
    """Routes Voice Agent server events to the renderer, player, and duplex state."""

    def __init__(
        self,
        *,
        renderer: Any,
        player: Any,
        full_duplex: bool = False,
        exit_after_reply: bool = False,
        ready_event: threading.Event | None = None,
    ) -> None:
        self.renderer = renderer
        self.player = player
        self.full_duplex = full_duplex
        # File-driven runs (`aai agent <file>`) stop after the agent's first reply
        # to the spoken input, so the process exits instead of waiting forever.
        self.exit_after_reply = exit_after_reply
        self.ready_event = ready_event
        # `ready`/`muted` are the duplex gate: written by dispatch() on the receive
        # loop, read by should_send_audio() on the capture thread. The lock makes that
        # cross-thread handoff explicit (and the reads consistent) instead of relying on
        # the GIL. `finished`/`_saw_user` are touched only by dispatch (one thread).
        self._lock = threading.Lock()
        self.ready = False
        self.muted = False
        self.finished = False
        self._saw_user = False

    def should_send_audio(self) -> bool:
        """True when captured mic frames should be forwarded to the server."""
        with self._lock:
            return self.ready and not self.muted

    def dispatch(self, event: dict[str, Any]) -> None:
        """Route one server event to its handler; unknown types are ignored.

        Handlers are registered in ``_EVENT_HANDLERS`` (below the class). Events with
        no handler — ``input.speech.stopped``, ``tool.call``, anything new — are no-ops.
        """
        handler = _EVENT_HANDLERS.get(event.get("type", ""))
        if handler is not None:
            handler(self, event)

    def _on_session_ready(self, _event: dict[str, Any]) -> None:
        with self._lock:
            self.ready = True
        if self.ready_event is not None:
            self.ready_event.set()
        self.renderer.connected()

    def _on_speech_started(self, _event: dict[str, Any]) -> None:
        if self.full_duplex:
            self.player.flush()

    def _on_user_delta(self, event: dict[str, Any]) -> None:
        self.renderer.user_partial(event.get("text", ""))

    def _on_user_final(self, event: dict[str, Any]) -> None:
        self._saw_user = True
        self.renderer.user_final(event.get("text", ""))

    def _on_reply_started(self, _event: dict[str, Any]) -> None:
        if not self.full_duplex:
            with self._lock:
                self.muted = True
        self.renderer.reply_started()

    def _on_reply_audio(self, event: dict[str, Any]) -> None:
        data = event.get("data")
        if data:
            self.player.enqueue(base64.b64decode(data))

    def _on_agent_transcript(self, event: dict[str, Any]) -> None:
        self.renderer.agent_transcript(
            event.get("text", ""), interrupted=bool(event.get("interrupted", False))
        )

    def _on_reply_done(self, event: dict[str, Any]) -> None:
        if not self.full_duplex:
            with self._lock:
                self.muted = False
        interrupted = event.get("status") == "interrupted"
        if interrupted:
            self.player.flush()
        self.renderer.reply_done(interrupted=interrupted)
        # File-driven run: the agent has answered the spoken input, so stop.
        if self.exit_after_reply and self._saw_user and not interrupted:
            self.finished = True

    def _raise_error(self, event: dict[str, Any]) -> None:
        code = event.get("code", "")
        message = event.get("message") or code or "Voice agent error."
        if code in _AUTH_ERROR_CODES:
            raise CLIError(
                f"Voice agent rejected the connection: {message}",
                error_type="unauthorized",
                exit_code=2,
            )
        raise APIError(f"Voice agent error ({code}): {message}")


# Server event type -> the VoiceAgentSession method that handles it. Types absent
# here (input.speech.stopped, tool.call, anything unrecognized) are ignored.
_EVENT_HANDLERS: dict[str, Callable[[VoiceAgentSession, dict[str, Any]], None]] = {
    "session.ready": VoiceAgentSession._on_session_ready,
    "input.speech.started": VoiceAgentSession._on_speech_started,
    "transcript.user.delta": VoiceAgentSession._on_user_delta,
    "transcript.user": VoiceAgentSession._on_user_final,
    "reply.started": VoiceAgentSession._on_reply_started,
    "reply.audio": VoiceAgentSession._on_reply_audio,
    "transcript.agent": VoiceAgentSession._on_agent_transcript,
    "reply.done": VoiceAgentSession._on_reply_done,
    "session.error": VoiceAgentSession._raise_error,
}


def _send_audio_loop(ws: Any, session: VoiceAgentSession, mic: Any) -> None:
    """Forward mic PCM as input.audio while the session gate allows it."""
    # File-driven runs wait for session.ready before consuming the source, so a
    # finite clip isn't partly drained (and dropped) before the server accepts it.
    if session.ready_event is not None:
        session.ready_event.wait(timeout=10)
    for chunk in mic:
        if not session.should_send_audio():
            continue  # half-duplex: drop frames while the agent is speaking
        payload = base64.b64encode(chunk).decode("ascii")
        try:
            ws.send(json.dumps({"type": "input.audio", "audio": payload}))
        except Exception:  # noqa: BLE001 - socket closed; capture thread ends
            return


def _auth_or_api_error(exc: Exception, message: str) -> CLIError:
    """Map a connect/session exception to the right CLIError: a rejected key becomes
    auth_failure(), anything else becomes APIError(f"{message}: {exc}")."""
    if is_auth_failure(exc):
        return auth_failure()
    return APIError(f"{message}: {exc}")


def _open_ws(connect: Any, api_key: str) -> Any:
    """Open the Voice Agent socket, mapping a connect failure to a clean CLIError."""
    try:
        return connect(_ws_url(), additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as exc:
        raise _auth_or_api_error(exc, "Could not connect to the voice agent") from exc


def run_session(
    api_key: str,
    *,
    renderer: Any,
    player: Any,
    mic: Any,
    voice: str,
    system_prompt: str,
    greeting: str,
    full_duplex: bool = False,
    exit_after_reply: bool = False,
    connect: Any = None,
) -> None:
    """Open the Voice Agent WebSocket and run the bidirectional loop until close.

    `connect` defaults to websockets' synchronous client; injectable for tests.
    When `exit_after_reply` is set (file-driven runs), the loop stops after the
    agent's first reply to the spoken input and the capture thread waits for
    session.ready before streaming the source.
    """
    if connect is None:
        from websockets.sync.client import connect

    ready_event = threading.Event() if exit_after_reply else None
    session = VoiceAgentSession(
        renderer=renderer,
        player=player,
        full_duplex=full_duplex,
        exit_after_reply=exit_after_reply,
        ready_event=ready_event,
    )

    ws = _open_ws(connect, api_key)

    # The mic opens lazily on first iteration, inside the capture thread; a failure
    # there (no device, sounddevice missing) must reach the user instead of vanishing
    # with the daemon thread. Capture it and close the socket to end the receive loop.
    # The append below happens-before the main thread reads `capture_error`: the
    # capture thread appends, then ws.close() ends the `for raw in ws` loop, so the
    # error is always visible by the time we inspect it after the loop.
    capture_error: list[CLIError] = []

    def _capture() -> None:
        try:
            _send_audio_loop(ws, session, mic)
        except CLIError as exc:
            capture_error.append(exc)
            with contextlib.suppress(Exception):
                ws.close()

    player_started = False
    try:
        player.start()  # opens the speaker stream; CLIError here if sounddevice can't load
        player_started = True
        threading.Thread(target=_capture, daemon=True).start()
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "system_prompt": system_prompt,
                        "greeting": greeting,
                        "output": {"voice": voice},
                    },
                }
            )
        )
        for raw in ws:
            session.dispatch(json.loads(raw))
            if session.finished:
                break
    except (CLIError, KeyboardInterrupt, BrokenPipeError):
        raise  # clean CLI errors, user Ctrl-C, and a closed pipe are handled upstream
    except Exception as exc:
        if capture_error:
            raise capture_error[0] from exc  # a mic-open failure is the real cause
        raise _auth_or_api_error(exc, "Voice agent session failed") from exc
    finally:
        with contextlib.suppress(Exception):
            ws.close()
        if player_started:
            player.close()
    # The receive loop can also end cleanly when the capture thread closes the
    # socket after a mic failure; surface that error rather than exiting 0.
    if capture_error:
        raise capture_error[0]
