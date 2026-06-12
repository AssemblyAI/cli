from __future__ import annotations

import base64
import contextlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aai_cli import environments
from aai_cli import ws as wsutil
from aai_cli.errors import APIError, CLIError, NotAuthenticated
from aai_cli.streaming import diagnostics


def ws_url() -> str:
    """Voice Agent socket URL for the active environment (set at CLI startup).

    Shared with code_gen so generated agent scripts target the same host the
    CLI itself would connect to.
    """
    return f"wss://{environments.active().agents_host}/v1/ws"


DEFAULT_PROMPT = (
    "You are a friendly voice assistant having a casual conversation. Keep replies "
    "short and natural, usually one or two sentences. Speak the way a person would "
    "in real conversation: relaxed, low-key, no exclamation marks."
)
DEFAULT_GREETING = "Hey, what's on your mind?"

# session.error codes that mean the connection is unauthorized -> exit 4, the same
# NotAuthenticated code every other rejected-credential path across the CLI uses.
_AUTH_ERROR_CODES = {"UNAUTHORIZED", "FORBIDDEN"}

# How long a file-driven run waits for the server's session.ready before giving up.
_READY_TIMEOUT_SECONDS = 10

# The websocket connection, the `connect` factory, and the renderer/player/mic I/O
# objects come from libraries/modules with no usable type stubs. Alias that untyped
# boundary here so each role is named in signatures and `Any` stays in one place.
# (Server event payloads remain `dict[str, Any]`.)
_WebSocket = Any
_Connect = Any
_IO = Any


@dataclass(frozen=True)
class AgentRunConfig:
    """The static (per-run) configuration for a Voice Agent session.

    Bundled into one value so `run_session`'s signature stays small: the I/O
    objects (renderer/player/mic) vary per call, but these knobs are fixed once
    the command has parsed its flags.
    """

    voice: str
    system_prompt: str
    greeting: str
    full_duplex: bool = False
    exit_after_reply: bool = False


class VoiceAgentSession:
    """Routes Voice Agent server events to the renderer, player, and duplex state."""

    def __init__(
        self,
        *,
        renderer: _IO,
        player: _IO,
        full_duplex: bool = False,
        exit_after_reply: bool = False,
        ready_event: threading.Event | None = None,
    ) -> None:
        self.renderer = renderer
        self.player = player
        self.full_duplex = full_duplex
        # File-driven runs (`assembly agent <file>`) stop after the agent's first reply
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

    def on_session_ready(self, _event: dict[str, Any]) -> None:
        with self._lock:
            self.ready = True
        if self.ready_event is not None:
            self.ready_event.set()
        self.renderer.connected()

    def on_speech_started(self, _event: dict[str, Any]) -> None:
        if self.full_duplex:
            self.player.flush()

    def on_user_delta(self, event: dict[str, Any]) -> None:
        self.renderer.user_partial(event.get("text", ""))

    def on_user_final(self, event: dict[str, Any]) -> None:
        self._saw_user = True
        self.renderer.user_final(event.get("text", ""))

    def on_reply_started(self, _event: dict[str, Any]) -> None:
        if not self.full_duplex:
            with self._lock:
                self.muted = True
        self.renderer.reply_started()

    def on_reply_audio(self, event: dict[str, Any]) -> None:
        data = event.get("data")
        if data:
            self.player.enqueue(base64.b64decode(data))

    def on_agent_transcript(self, event: dict[str, Any]) -> None:
        self.renderer.agent_transcript(
            event.get("text", ""), interrupted=bool(event.get("interrupted", False))
        )

    def on_reply_done(self, event: dict[str, Any]) -> None:
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

    def raise_error(self, event: dict[str, Any]) -> None:
        code = event.get("code", "")
        message = event.get("message") or code or "Voice agent error."
        if code in _AUTH_ERROR_CODES:
            # rejected_key: a credential was presented and the server refused it, so
            # auto-login must not retry when the key came from ASSEMBLYAI_API_KEY.
            raise NotAuthenticated(
                f"Voice agent rejected the connection: {message}",
                suggestion="Run 'assembly login' with a valid key, or set ASSEMBLYAI_API_KEY.",
                rejected_key=True,
            )
        raise APIError(f"Voice agent error ({code}): {message}")


# Server event type -> the VoiceAgentSession method that handles it. Types absent
# here (input.speech.stopped, tool.call, anything unrecognized) are ignored.
_EVENT_HANDLERS: dict[str, Callable[[VoiceAgentSession, dict[str, Any]], None]] = {
    "session.ready": VoiceAgentSession.on_session_ready,
    "input.speech.started": VoiceAgentSession.on_speech_started,
    "transcript.user.delta": VoiceAgentSession.on_user_delta,
    "transcript.user": VoiceAgentSession.on_user_final,
    "reply.started": VoiceAgentSession.on_reply_started,
    "reply.audio": VoiceAgentSession.on_reply_audio,
    "transcript.agent": VoiceAgentSession.on_agent_transcript,
    "reply.done": VoiceAgentSession.on_reply_done,
    "session.error": VoiceAgentSession.raise_error,
}


def _send_audio_loop(ws: _WebSocket, session: VoiceAgentSession, mic: _IO) -> None:
    """Forward mic PCM as input.audio while the session gate allows it."""
    # File-driven runs wait for session.ready before consuming the source, so a
    # finite clip isn't partly drained (and dropped) before the server accepts it.
    # A server that never becomes ready fails the run loudly: continuing would
    # silently discard the start of the clip frame by frame.
    if session.ready_event is not None and not session.ready_event.wait(
        timeout=_READY_TIMEOUT_SECONDS
    ):
        raise APIError(
            f"Voice agent session was not ready after {_READY_TIMEOUT_SECONDS}s; "
            "no audio was sent.",
            suggestion="Try again; if it persists, check https://status.assemblyai.com.",
        )
    for chunk in mic:
        if not session.should_send_audio():
            continue  # half-duplex: drop frames while the agent is speaking
        payload = base64.b64encode(chunk).decode("ascii")
        try:
            ws.send(json.dumps({"type": "input.audio", "audio": payload}))
        except Exception:  # noqa: BLE001 - socket closed; capture thread ends
            return


def _open_ws(connect: _Connect, api_key: str) -> _WebSocket:
    """Open the Voice Agent socket, mapping a connect failure to a clean CLIError.

    A rejected handshake (HTTP 401/403) gets the shared actionable suggestion
    (whoami / environment / network); anything else keeps the wsutil mapping.
    """
    message = "Could not connect to the voice agent"
    try:
        return connect(ws_url(), additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as exc:
        rejected = diagnostics.handshake_error(exc, message, host=environments.active().agents_host)
        if rejected is not None:
            raise rejected from exc
        raise wsutil.auth_or_api_error(exc, message) from exc


def _session_update_message(config: AgentRunConfig) -> str:
    """The initial session.update payload as a JSON string: persona, greeting, voice."""
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "system_prompt": config.system_prompt,
                "greeting": config.greeting,
                "output": {"voice": config.voice},
            },
        }
    )


def _receive_loop(ws: _WebSocket, session: VoiceAgentSession) -> None:
    """Dispatch inbound server events until the socket closes or the run finishes."""
    for raw in ws:
        session.dispatch(json.loads(raw))
        if session.finished:
            break


def run_session(
    api_key: str,
    *,
    renderer: _IO,
    player: _IO,
    mic: _IO,
    config: AgentRunConfig,
    connect: _Connect = None,
) -> None:
    """Open the Voice Agent WebSocket and run the bidirectional loop until close.

    `connect` defaults to websockets' synchronous client; injectable for tests.
    When `config.exit_after_reply` is set (file-driven runs), the loop stops after
    the agent's first reply to the spoken input and the capture thread waits for
    session.ready before streaming the source.
    """
    wsutil.silence_websockets_logging()
    if connect is None:
        from websockets.sync.client import connect

    ready_event = threading.Event() if config.exit_after_reply else None
    session = VoiceAgentSession(
        renderer=renderer,
        player=player,
        full_duplex=config.full_duplex,
        exit_after_reply=config.exit_after_reply,
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
        ws.send(_session_update_message(config))
        _receive_loop(ws, session)
    except (CLIError, KeyboardInterrupt, BrokenPipeError):
        raise  # clean CLI errors, user Ctrl-C, and a closed pipe are handled upstream
    except Exception as exc:
        if capture_error:
            raise capture_error[0] from exc  # a mic-open failure is the real cause
        raise wsutil.auth_or_api_error(exc, "Voice agent session failed") from exc
    finally:
        with contextlib.suppress(Exception):
            ws.close()
        if player_started:
            player.close()
    # The receive loop can also end cleanly when the capture thread closes the
    # socket after a mic failure; surface that error rather than exiting 0.
    if capture_error:
        raise capture_error[0]
