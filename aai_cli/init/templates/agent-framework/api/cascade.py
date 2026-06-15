"""Server-side cascade orchestrator for the agent-framework template.

The browser opens one WebSocket to FastAPI and the backend wires three AssemblyAI
primitives together — Streaming STT, the LLM Gateway, and streaming TTS — so every
credential stays on the server. The orchestrator takes injected connect-factories and
an LLM callable (`Deps`) so it runs hermetically against fakes in tests, the same
seam `aai_cli/tts/session.py` uses.

Browser protocol (identical to the voice-agent template):
  in : {"type": "input.audio", "audio": <base64 PCM>}
  out: transcript.user / transcript.agent / reply.audio (base64 in `data`) /
       input.speech.started / reply.done / session.error
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode


def unavailable_reason(settings: Any) -> str | None:
    """Why the cascade can't run, or None when it can.

    Streaming TTS has no production host, so an empty TTS host means the user must
    re-scaffold against the sandbox.
    """
    if not settings.API_KEY:
        return "ASSEMBLYAI_API_KEY is not set — configure it in your deployment's environment."
    if not settings.TTS_HOST:
        return (
            "Streaming TTS has no production host, so this cascade is sandbox-only. "
            "Re-scaffold against the sandbox: assembly --sandbox init agent-framework."
        )
    return None


def stt_url(settings: Any) -> str:
    """The Streaming v3 WebSocket URL with PCM + turn-formatting params."""
    params = urlencode(
        {
            "sample_rate": settings.INPUT_SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "speech_model": "u3-rt-pro",
            "format_turns": "true",
        }
    )
    return f"wss://{settings.STREAMING_HOST}/v3/ws?{params}"


def tts_url(settings: Any) -> str:
    """The streaming-TTS WebSocket URL for the configured voice and sample rate."""
    params = urlencode({"voice": settings.VOICE, "sample_rate": settings.OUTPUT_SAMPLE_RATE})
    return f"wss://{settings.TTS_HOST}/v1/ws/?{params}"


def is_final_user_turn(msg: dict[str, Any]) -> bool:
    """True for a finalized, formatted end-of-turn (the cue to reply)."""
    return bool(msg.get("end_of_turn")) and bool(msg.get("turn_is_formatted"))


def build_messages(system_prompt: str, user_text: str) -> list[dict[str, str]]:
    """The chat `messages` array for one user turn."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


@dataclass
class Deps:
    """Injected cascade dependencies. `Deps.real(settings)` wires the live clients;
    tests pass fakes with the same shapes."""

    connect_stt: Callable[[], Awaitable[Any]]
    connect_tts: Callable[[], Awaitable[Any]]
    llm_stream: Callable[[list[dict[str, str]]], AsyncIterator[str]]
    settings: Any

    @classmethod
    def real(cls, settings: Any) -> Deps:
        return cls(
            connect_stt=lambda: _connect_stt(settings),
            connect_tts=lambda: _connect_tts(settings),
            llm_stream=lambda messages: _llm_stream(settings, messages),
            settings=settings,
        )


class Session:
    """Tracks the in-flight reply so a new user turn can barge in and cancel it."""

    def __init__(self) -> None:
        self.reply_task: asyncio.Task[None] | None = None

    async def cancel_reply(self) -> None:
        task, self.reply_task = self.reply_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def drain(self) -> None:
        """Await the in-flight reply to natural completion (used when STT closes)."""
        task = self.reply_task
        if task is not None:
            with contextlib.suppress(Exception):
                await task


async def _connect_stt(settings: Any) -> Any:
    import websockets

    return await websockets.connect(
        stt_url(settings), additional_headers={"Authorization": settings.API_KEY}
    )


async def _connect_tts(settings: Any) -> Any:
    import websockets

    # max_size=None: a synthesis's Audio frames can exceed the 1 MiB default.
    return await websockets.connect(
        tts_url(settings),
        additional_headers={"Authorization": settings.API_KEY},
        max_size=None,
    )


async def _llm_stream(settings: Any, messages: list[dict[str, str]]) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.LLM_GATEWAY_URL, api_key=settings.API_KEY)
    stream = await client.chat.completions.create(
        model=settings.MODEL, messages=messages, stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _safe_close(conn: Any) -> None:
    with contextlib.suppress(Exception):
        await conn.close()


async def _pump_mic(browser: Any, stt: Any) -> None:
    """Forward each base64 mic frame from the browser to the STT socket."""
    while True:
        msg = await browser.recv()
        if msg is None:
            return
        audio = msg.get("audio") if msg.get("type") == "input.audio" else None
        if isinstance(audio, str):
            await stt.send(base64.b64decode(audio))


async def _synthesize(browser: Any, tts: Any, text: str) -> None:
    """Drive the TTS protocol on an open socket, forwarding Audio as reply.audio."""
    begin = json.loads(await tts.recv())
    if begin.get("type") != "Begin":
        raise RuntimeError(f"TTS did not begin (got {begin.get('type')!r}).")
    await tts.send(json.dumps({"type": "Generate", "text": text}))
    await tts.send(json.dumps({"type": "ForceFlushTextBuffer"}))
    while True:
        frame = json.loads(await tts.recv())
        kind = frame.get("type")
        if kind == "Audio":
            await browser.send({"type": "reply.audio", "data": frame.get("audio", "")})
            if frame.get("is_final"):
                break
        elif kind == "Error":
            raise RuntimeError(frame.get("error") or "TTS error")
    with contextlib.suppress(Exception):
        await tts.send(json.dumps({"type": "Terminate"}))
    await _safe_close(tts)


async def _speak(browser: Any, deps: Deps, text: str) -> None:
    """Emit agent text, synthesize it, and mark the reply done."""
    await browser.send({"type": "transcript.agent", "text": text})
    tts = await deps.connect_tts()
    try:
        await _synthesize(browser, tts, text)
    finally:
        await _safe_close(tts)
    await browser.send({"type": "reply.done", "status": "completed"})


async def _generate_reply(browser: Any, deps: Deps, messages: list[dict[str, str]]) -> None:
    """Stream the LLM reply, then speak it. Errors surface as session.error."""
    try:
        text = "".join([delta async for delta in deps.llm_stream(messages)]).strip()
        if not text:
            await browser.send({"type": "reply.done", "status": "empty"})
            return
        await _speak(browser, deps, text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # any leg failure becomes one clean session.error event
        await browser.send({"type": "session.error", "message": str(exc)})


async def maybe_barge_in(browser: Any, session: Session) -> None:
    """If a reply is playing, tell the browser to stop and cancel it."""
    if session.reply_task is not None and not session.reply_task.done():
        await browser.send({"type": "input.speech.started"})
        await session.cancel_reply()


async def _pump_stt(browser: Any, stt: Any, deps: Deps, session: Session) -> None:
    """Read STT turns: emit user transcripts, reply on finalized turns, barge in on
    interim speech, and drain the last reply when the socket closes."""
    async for raw in stt:
        msg = json.loads(raw)
        if msg.get("type") != "Turn":
            continue
        text = msg.get("transcript", "")
        if not text:
            continue
        await browser.send({"type": "transcript.user", "text": text})
        if is_final_user_turn(msg):
            await session.cancel_reply()
            session.reply_task = asyncio.create_task(
                _generate_reply(browser, deps, build_messages(deps.settings.SYSTEM_PROMPT, text))
            )
        else:
            await maybe_barge_in(browser, session)
    await session.drain()


async def run_session(browser: Any, deps: Deps) -> None:
    """Run one browser session: greet, then cascade STT -> LLM -> TTS until either
    side closes. All credentials stay server-side."""
    reason = unavailable_reason(deps.settings)
    if reason is not None:
        await browser.send({"type": "session.error", "message": reason})
        return
    try:
        stt = await deps.connect_stt()
    except Exception as exc:  # any connect/setup failure becomes one clean session.error
        await browser.send(
            {"type": "session.error", "message": f"Could not start the session: {exc}"}
        )
        return

    session = Session()
    session.reply_task = asyncio.create_task(_speak(browser, deps, deps.settings.GREETING))
    mic = asyncio.create_task(_pump_mic(browser, stt))
    listen = asyncio.create_task(_pump_stt(browser, stt, deps, session))
    try:
        await asyncio.wait({mic, listen}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        mic.cancel()
        listen.cancel()
        await asyncio.gather(mic, listen, return_exceptions=True)
        await session.cancel_reply()
        await _safe_close(stt)


class FastAPIBrowser:
    """Adapts a Starlette WebSocket to the (send, recv) shape run_session expects.
    recv() returns None when the client disconnects, so the pumps exit cleanly."""

    def __init__(self, websocket: Any) -> None:
        self._ws = websocket

    async def send(self, event: dict[str, Any]) -> None:
        await self._ws.send_json(event)

    async def recv(self) -> dict[str, Any] | None:
        from fastapi import WebSocketDisconnect

        try:
            return await self._ws.receive_json()
        except WebSocketDisconnect:
            return None
