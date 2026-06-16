"""Hermetic tests for the agent-cascade template's conversation memory + multi-turn STT.

Covers the sliding-window history threaded through `_generate_reply` and `_pump_stt`:
the user turn is recorded before the reply, the assistant turn after it completes, so a
multi-turn session accumulates an alternating user/assistant transcript.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from tests._agent_cascade import FakeBrowser, FakeWS, _async_return, _cascade, _deps, reimport


def test_generate_reply_records_history(monkeypatch):
    # Seed a prior exchange; the reply must thread it through build_messages -> llm_stream
    # AND append the new assistant turn.
    captured: list[list[dict[str, str]]] = []
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {}},
            {"type": "Audio", "audio": "AAA=", "is_final": True},
        ],
        llm_text=["Sure."],
        captured_messages=captured,
    )
    browser = FakeBrowser()
    session = cascade.Session()
    session.history = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    asyncio.run(cascade._generate_reply(browser, deps, session))
    assert captured[0] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "earlier reply"},
    ]
    assert session.history[-1] == {"role": "assistant", "content": "Sure."}


def test_generate_reply_speaks_unpunctuated_tail(monkeypatch):
    # An LLM reply with no terminal punctuation leaves a non-empty tail after the loop;
    # that remainder must still be spoken (its own TTS socket) and recorded.
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {}},
            {"type": "Audio", "audio": "AAA=", "is_final": True},
        ],
        llm_text=["no period here"],
    )
    browser = FakeBrowser()
    session = cascade.Session()
    asyncio.run(cascade._generate_reply(browser, deps, session))
    agent_texts = [e["text"] for e in browser.sent if e["type"] == "transcript.agent"]
    assert agent_texts == ["no period here"]
    assert {"type": "reply.audio", "data": "AAA="} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}
    assert session.history[-1] == {"role": "assistant", "content": "no period here"}


def _turn(text: str) -> dict[str, object]:
    return {
        "type": "Turn",
        "transcript": text,
        "end_of_turn": True,
        "turn_is_formatted": True,
    }


class _DrainingSTT(FakeWS):
    """An STT fake that lets the prior reply task finish before emitting the next turn.

    `_pump_stt` schedules each reply as a background task; a second turn arriving while
    the first reply is mid-flight would cancel (barge in on) it. To exercise the
    *uninterrupted* multi-turn path — where each reply completes and appends its
    assistant turn — this runs the injected drain callback (which awaits the session's
    in-flight reply) before yielding the next turn.
    """

    def __init__(
        self, drain: Callable[[], Awaitable[None]], frames: list[dict[str, object]]
    ) -> None:
        super().__init__(frames)
        self._drain = drain

    async def __anext__(self) -> str:
        await self._drain()
        return await super().__anext__()


def test_pump_stt_accumulates_multi_turn_history(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two finalized turns, each reply allowed to complete, must accumulate history to
    # [user1, assistant1, user2, assistant2] in order.
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.SYSTEM_PROMPT = "be brief"
    browser = FakeBrowser()

    replies = iter(["First reply.", "Second reply."])

    async def llm_stream(_messages):
        yield next(replies)

    async def connect_tts():
        return FakeWS(
            [
                {"type": "Begin", "configuration": {}},
                {"type": "Audio", "audio": "AAA=", "is_final": True},
            ]
        )

    session = cascade.Session()

    async def drain_reply() -> None:
        if session.reply_task is not None:
            await session.reply_task

    stt = _DrainingSTT(drain_reply, [_turn("hello"), _turn("again")])
    deps = cascade.Deps(
        connect_stt=_async_return(stt),
        connect_tts=connect_tts,
        llm_stream=llm_stream,
        settings=settings,
    )
    asyncio.run(asyncio.wait_for(cascade._pump_stt(browser, stt, deps, session), timeout=5))
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "First reply."},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "Second reply."},
    ]
