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
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlencode

from fastapi import WebSocket

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from websockets.asyncio.client import ClientConnection


class _Settings(Protocol):
    API_KEY: str
    STREAMING_HOST: str
    TTS_HOST: str
    LLM_GATEWAY_URL: str
    MODEL: str
    VOICE: str
    SYSTEM_PROMPT: str
    GREETING: str
    MAX_HISTORY: int
    INPUT_SAMPLE_RATE: int
    OUTPUT_SAMPLE_RATE: int


class _Browser(Protocol):
    async def send(self, event: dict[str, object]) -> None:
        """Send one protocol event to the browser."""

    async def recv(self) -> dict[str, object] | None:
        """Receive the next browser message, or None once the socket closes."""


def unavailable_reason(settings: _Settings) -> str | None:
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


def stt_url(settings: _Settings) -> str:
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


def tts_url(settings: _Settings) -> str:
    """The streaming-TTS WebSocket URL for the configured voice and sample rate."""
    params = urlencode({"voice": settings.VOICE, "sample_rate": settings.OUTPUT_SAMPLE_RATE})
    return f"wss://{settings.TTS_HOST}/v1/ws/?{params}"


def is_final_user_turn(msg: dict[str, object]) -> bool:
    """True for a finalized, formatted end-of-turn (the cue to reply)."""
    return bool(msg.get("end_of_turn")) and bool(msg.get("turn_is_formatted"))


def build_messages(
    system_prompt: str, history: list[ChatCompletionMessageParam]
) -> list[ChatCompletionMessageParam]:
    """The chat `messages` array: the system prompt followed by the conversation so far."""
    return [{"role": "system", "content": system_prompt}, *history]


def _trim_history(history: list[ChatCompletionMessageParam], max_messages: int) -> None:
    """Cap the running history to the most recent ``max_messages`` (sliding window)."""
    if len(history) > max_messages:
        del history[: len(history) - max_messages]


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off complete sentences (each ending in . ! ?). Return (sentences, remainder)."""
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(buffer):
        if char in ".!?":
            sentence = buffer[start : index + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = index + 1
    return sentences, buffer[start:]


@dataclass
class Deps:
    """Injected cascade dependencies. `Deps.real(settings)` wires the live clients;
    tests pass fakes with the same shapes."""

    connect_stt: Callable[[], Awaitable[ClientConnection]]
    connect_tts: Callable[[], Awaitable[ClientConnection]]
    llm_stream: Callable[[list[ChatCompletionMessageParam]], AsyncIterator[str]]
    settings: _Settings

    @classmethod
    def real(cls, settings: _Settings) -> Deps:
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
        self.history: list[ChatCompletionMessageParam] = []

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


async def _connect_stt(settings: _Settings) -> ClientConnection:
    import websockets

    return await websockets.connect(
        stt_url(settings), additional_headers={"Authorization": settings.API_KEY}
    )


async def _connect_tts(settings: _Settings) -> ClientConnection:
    import websockets

    # max_size=None: a synthesis's Audio frames can exceed the 1 MiB default.
    return await websockets.connect(
        tts_url(settings),
        additional_headers={"Authorization": settings.API_KEY},
        max_size=None,
    )


async def _llm_stream(
    settings: _Settings, messages: list[ChatCompletionMessageParam]
) -> AsyncIterator[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=settings.LLM_GATEWAY_URL, api_key=settings.API_KEY)
    stream = await client.chat.completions.create(
        model=settings.MODEL, messages=messages, stream=True
    )
    async for chunk in stream:
        # The gateway (Anthropic-backed, OpenAI-compatible) ends the stream with a
        # usage/final chunk that carries no choices — skip it instead of IndexError-ing.
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _safe_close(conn: ClientConnection) -> None:
    with contextlib.suppress(Exception):
        await conn.close()


async def _pump_mic(browser: _Browser, stt: ClientConnection) -> None:
    """Forward each base64 mic frame from the browser to the STT socket."""
    while True:
        msg = await browser.recv()
        if msg is None:
            return
        audio = msg.get("audio") if msg.get("type") == "input.audio" else None
        if isinstance(audio, str):
            await stt.send(base64.b64decode(audio))


async def _synthesize(browser: _Browser, tts: ClientConnection, text: str) -> None:
    """Drive the TTS protocol on an open socket, forwarding Audio as reply.audio."""
    begin = json.loads(await tts.recv())
    if begin.get("type") != "Begin":
        raise RuntimeError(f"TTS did not begin (got {begin.get('type')!r}).")
    await tts.send(json.dumps({"type": "Generate", "text": text}))
    await tts.send(json.dumps({"type": "Flush"}))
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


async def _speak(browser: _Browser, deps: Deps, text: str) -> None:
    """Emit agent text, synthesize it, and mark the reply done."""
    await browser.send({"type": "transcript.agent", "text": text})
    tts = await deps.connect_tts()
    try:
        await _synthesize(browser, tts, text)
    finally:
        await _safe_close(tts)
    await browser.send({"type": "reply.done", "status": "completed"})


async def _speak_sentence(browser: _Browser, deps: Deps, text: str) -> None:
    """Show + synthesize one sentence of a streamed reply (no reply.done)."""
    await browser.send({"type": "transcript.agent", "text": text})
    tts = await deps.connect_tts()
    try:
        await _synthesize(browser, tts, text)
    finally:
        await _safe_close(tts)


async def _generate_reply(browser: _Browser, deps: Deps, session: Session) -> None:
    """Stream the LLM reply sentence-by-sentence into TTS (low perceived latency), then
    record it in the conversation history. Errors surface as session.error."""
    messages = build_messages(deps.settings.SYSTEM_PROMPT, session.history)
    try:
        spoken: list[str] = []
        buffer = ""
        async for delta in deps.llm_stream(messages):
            buffer += delta
            sentences, buffer = _split_sentences(buffer)
            for sentence in sentences:
                spoken.append(sentence)
                await _speak_sentence(browser, deps, sentence)
        tail = buffer.strip()
        if tail:
            spoken.append(tail)
            await _speak_sentence(browser, deps, tail)
        reply = " ".join(spoken).strip()
        if not reply:
            await browser.send({"type": "reply.done", "status": "empty"})
            return
        session.history.append({"role": "assistant", "content": reply})
        _trim_history(session.history, deps.settings.MAX_HISTORY)
        await browser.send({"type": "reply.done", "status": "completed"})
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # any leg failure becomes one clean session.error event
        await browser.send({"type": "session.error", "message": str(exc)})


async def maybe_barge_in(browser: _Browser, session: Session) -> None:
    """If a reply is playing, tell the browser to stop and cancel it."""
    if session.reply_task is not None and not session.reply_task.done():
        await browser.send({"type": "input.speech.started"})
        await session.cancel_reply()


async def _pump_stt(browser: _Browser, stt: ClientConnection, deps: Deps, session: Session) -> None:
    """Read STT turns: display only the finalized (formatted end-of-turn) user
    transcript and reply to it. An interim turn isn't shown — it only barges in on a
    playing reply. Drain the last reply when the socket closes."""
    async for raw in stt:
        msg = json.loads(raw)
        if msg.get("type") != "Turn":
            continue
        text = msg.get("transcript", "")
        if not text:
            continue
        if is_final_user_turn(msg):
            await browser.send({"type": "transcript.user", "text": text})
            await session.cancel_reply()
            session.history.append({"role": "user", "content": text})
            _trim_history(session.history, deps.settings.MAX_HISTORY)
            session.reply_task = asyncio.create_task(_generate_reply(browser, deps, session))
        else:
            await maybe_barge_in(browser, session)
    await session.drain()


class _SessionClosed(Exception):
    """Sentinel that unwinds the session TaskGroup when one pump returns — i.e. the
    browser disconnected or the STT socket closed. Raising it cancels the sibling pump."""


async def _until_closed(pump: Awaitable[None]) -> None:
    """Run a pump to its natural end, then raise to close the session TaskGroup."""
    await pump
    raise _SessionClosed


async def run_session(browser: _Browser, deps: Deps) -> None:
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
    try:
        # Race the two pumps: whichever returns first (browser hangs up → mic; STT
        # socket closes → listen) raises _SessionClosed, and the TaskGroup cancels the
        # other pump for us — no manual cancel/gather bookkeeping.
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_until_closed(_pump_mic(browser, stt)))
            tg.create_task(_until_closed(_pump_stt(browser, stt, deps, session)))
    except* _SessionClosed:
        pass
    finally:
        await session.cancel_reply()
        await _safe_close(stt)


class FastAPIBrowser:
    """Adapts a Starlette WebSocket to the (send, recv) shape run_session expects.
    recv() returns None when the client disconnects, so the pumps exit cleanly."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send(self, event: dict[str, object]) -> None:
        await self._ws.send_json(event)

    async def recv(self) -> dict[str, object] | None:
        from fastapi import WebSocketDisconnect

        try:
            data: dict[str, object] = await self._ws.receive_json()
        except WebSocketDisconnect:
            return None
        return data
