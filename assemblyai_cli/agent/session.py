from __future__ import annotations

import base64
import contextlib
import json
import threading
from typing import Any

from assemblyai_cli.errors import APIError, CLIError, auth_failure, is_auth_failure

WS_URL = "wss://agents.assemblyai.com/v1/ws"

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

    def __init__(self, *, renderer: Any, player: Any, full_duplex: bool = False) -> None:
        self.renderer = renderer
        self.player = player
        self.full_duplex = full_duplex
        self.ready = False
        self.muted = False

    def should_send_audio(self) -> bool:
        """True when captured mic frames should be forwarded to the server."""
        return self.ready and not self.muted

    def dispatch(self, event: dict) -> None:
        etype = event.get("type")

        if etype == "session.ready":
            self.ready = True
            self.renderer.connected()
        elif etype == "input.speech.started":
            if self.full_duplex:
                self.player.flush()
        elif etype == "input.speech.stopped":
            pass
        elif etype == "transcript.user.delta":
            self.renderer.user_partial(event.get("text", ""))
        elif etype == "transcript.user":
            self.renderer.user_final(event.get("text", ""))
        elif etype == "reply.started":
            if not self.full_duplex:
                self.muted = True
            self.renderer.reply_started()
        elif etype == "reply.audio":
            data = event.get("data")
            if data:
                self.player.enqueue(base64.b64decode(data))
        elif etype == "transcript.agent":
            self.renderer.agent_transcript(
                event.get("text", ""), interrupted=bool(event.get("interrupted", False))
            )
        elif etype == "reply.done":
            if not self.full_duplex:
                self.muted = False
            interrupted = event.get("status") == "interrupted"
            if interrupted:
                self.player.flush()
            self.renderer.reply_done(interrupted=interrupted)
        elif etype == "session.error":
            self._raise_error(event)
        # tool.call and unknown event types: intentionally ignored.

    def _raise_error(self, event: dict) -> None:
        code = event.get("code", "")
        message = event.get("message") or code or "Voice agent error."
        if code in _AUTH_ERROR_CODES:
            raise CLIError(
                f"Voice agent rejected the connection: {message}",
                error_type="unauthorized",
                exit_code=2,
            )
        raise APIError(f"Voice agent error ({code}): {message}")


def _send_audio_loop(ws: Any, session: VoiceAgentSession, mic: Any) -> None:
    """Forward mic PCM as input.audio while the session gate allows it."""
    for chunk in mic:
        if not session.should_send_audio():
            continue  # half-duplex: drop frames while the agent is speaking
        payload = base64.b64encode(chunk).decode("ascii")
        try:
            ws.send(json.dumps({"type": "input.audio", "audio": payload}))
        except Exception:  # noqa: BLE001 - socket closed; capture thread ends
            return


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
    connect: Any = None,
) -> None:
    """Open the Voice Agent WebSocket and run the bidirectional loop until close.

    `connect` defaults to websockets' synchronous client; injectable for tests.
    """
    _connect = connect
    if _connect is None:
        from websockets.sync.client import connect as _connect

    session = VoiceAgentSession(renderer=renderer, player=player, full_duplex=full_duplex)

    try:
        ws = _connect(WS_URL, additional_headers={"Authorization": f"Bearer {api_key}"})
    except Exception as exc:
        if is_auth_failure(exc):
            raise auth_failure() from exc
        raise APIError(f"Could not connect to the voice agent: {exc}") from exc

    player_started = False
    try:
        player.start()  # opens the speaker stream; CLIError here if PyAudio can't load
        player_started = True
        capture = threading.Thread(target=_send_audio_loop, args=(ws, session, mic), daemon=True)
        capture.start()
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
    except (CLIError, KeyboardInterrupt):
        raise  # auth/protocol errors and user Ctrl-C handled upstream
    except Exception as exc:
        if is_auth_failure(exc):
            # The Voice Agent server closes with 1008 (policy violation) on a bad key.
            raise auth_failure() from exc
        raise APIError(f"Voice agent session failed: {exc}") from exc
    finally:
        with contextlib.suppress(Exception):
            ws.close()
        if player_started:
            player.close()
