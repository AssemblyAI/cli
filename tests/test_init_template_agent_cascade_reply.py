"""Reply-synthesis-path tests for the agent-cascade template.

Covers _synthesize (the TTS protocol), _speak (the greeting/single-shot reply), and
_generate_reply (the streamed, sentence-by-sentence reply with conversation memory).
Split out of test_init_template_agent_cascade.py to keep each file under the
500-line gate.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests._agent_cascade import (
    FakeBrowser,
    FakeWS,
    _async_return,
    _cascade,
    _deps,
    reimport,
)


def test_synthesize_streams_audio_frames(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    tts = FakeWS(
        [
            {"type": "Begin", "configuration": {"sample_rate": 24000}},
            {"type": "Audio", "audio": "AAA="},
            {"type": "Audio", "audio": "BBB=", "is_final": True},
        ]
    )
    asyncio.run(cascade._synthesize(browser, tts, "hi"))
    assert browser.sent == [
        {"type": "reply.audio", "data": "AAA="},
        {"type": "reply.audio", "data": "BBB="},
    ]
    kinds = [json.loads(s)["type"] for s in tts.sent]
    assert kinds == ["Generate", "Flush", "Terminate"]
    # The Generate frame carries the text.
    assert json.loads(tts.sent[0])["text"] == "hi"
    assert tts.closed is True


def test_synthesize_raises_on_error_frame(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    tts = FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Error", "error": "bad voice"}])
    with pytest.raises(RuntimeError, match="bad voice"):
        asyncio.run(cascade._synthesize(browser, tts, "hi"))


def test_synthesize_raises_when_no_begin(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    tts = FakeWS([{"type": "Audio", "audio": "AAA=", "is_final": True}])
    with pytest.raises(RuntimeError, match="did not begin"):
        asyncio.run(cascade._synthesize(browser, tts, "hi"))


def test_synthesize_handles_close_without_final_frame(monkeypatch):
    # The TTS socket closes after some audio but before an is_final frame: the loop
    # must end cleanly (forward what arrived, then Terminate) instead of raising.
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    tts = FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Audio", "audio": "AAA="}])
    asyncio.run(cascade._synthesize(browser, tts, "hi"))
    assert {"type": "reply.audio", "data": "AAA="} in browser.sent
    assert json.loads(tts.sent[-1])["type"] == "Terminate"  # graceful teardown still ran
    assert tts.closed is True


def test_speak_emits_done_on_success(monkeypatch):
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[{"type": "Begin", "configuration": {}}, {"type": "Audio", "is_final": True}],
        llm_text=[],
    )
    browser = FakeBrowser()
    asyncio.run(cascade._speak(browser, deps, "hello!"))
    assert {"type": "transcript.agent", "text": "hello!"} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}


def test_speak_surfaces_error_instead_of_silent_failure(monkeypatch):
    # A greeting/TTS failure must become a clean session.error — not a swallowed
    # task exception that leaves the user with no audio and no signal.
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"

    async def failing_connect_tts():
        raise RuntimeError("tts unreachable")

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=failing_connect_tts,
        llm_stream=lambda _m: iter(()),
        settings=settings,
    )
    browser = FakeBrowser()
    asyncio.run(cascade._speak(browser, deps, "hello!"))
    assert browser.sent == [
        {"type": "transcript.agent", "text": "hello!"},
        {"type": "session.error", "message": "tts unreachable"},
    ]


def test_speak_propagates_cancellation(monkeypatch):
    # Barge-in on the greeting must cancel cleanly (re-raise), not become a session.error.
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"

    async def blocking_connect_tts():
        await asyncio.Event().wait()  # never resolves -> task blocks until cancelled
        return FakeWS()

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=blocking_connect_tts,
        llm_stream=lambda _m: iter(()),
        settings=settings,
    )
    browser = FakeBrowser()

    async def drive():
        task = asyncio.create_task(cascade._speak(browser, deps, "hello!"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(task)

    asyncio.run(drive())
    assert not any(e["type"] == "session.error" for e in browser.sent)


def test_generate_reply_speaks_llm_text(monkeypatch):
    # A single sentence: one TTS socket, one transcript.agent + reply.audio, one done.
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {}},
            {"type": "Audio", "audio": "AAA=", "is_final": True},
        ],
        llm_text=["Hello", " world."],
    )
    browser = FakeBrowser()
    session = cascade.Session()
    asyncio.run(cascade._generate_reply(browser, deps, session))
    assert {"type": "transcript.agent", "text": "Hello world."} in browser.sent
    assert {"type": "reply.audio", "data": "AAA="} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}
    assert session.history[-1] == {"role": "assistant", "content": "Hello world."}


def test_generate_reply_streams_each_sentence(monkeypatch):
    # Deltas form TWO sentences -> two TTS sockets, two transcript.agent + reply.audio.
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {}},
            {"type": "Audio", "audio": "AAA=", "is_final": True},
        ],
        llm_text=["Hello there. ", "How are you?"],
    )
    browser = FakeBrowser()
    session = cascade.Session()
    asyncio.run(cascade._generate_reply(browser, deps, session))
    agent_texts = [e["text"] for e in browser.sent if e["type"] == "transcript.agent"]
    assert agent_texts == ["Hello there.", "How are you?"]
    audio = [e for e in browser.sent if e["type"] == "reply.audio"]
    assert audio == [{"type": "reply.audio", "data": "AAA="}] * 2
    done = [e for e in browser.sent if e["type"] == "reply.done"]
    assert done == [{"type": "reply.done", "status": "completed"}]
    assert session.history[-1] == {"role": "assistant", "content": "Hello there. How are you?"}


def test_generate_reply_empty_llm_emits_done(monkeypatch):
    cascade, deps = _deps(monkeypatch, stt=FakeWS(), tts_frames=[], llm_text=["  "])
    browser = FakeBrowser()
    session = cascade.Session()
    asyncio.run(cascade._generate_reply(browser, deps, session))
    assert browser.sent == [{"type": "reply.done", "status": "empty"}]
    assert session.history == []  # nothing recorded for an empty reply


def test_generate_reply_surfaces_error(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"

    async def llm_stream(_messages):
        yield "partial"  # one delta arrives (no boundary), then the LLM leg fails mid-stream
        raise RuntimeError("llm down")

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=_async_return(FakeWS()),
        llm_stream=llm_stream,
        settings=settings,
    )
    browser = FakeBrowser()
    asyncio.run(cascade._generate_reply(browser, deps, cascade.Session()))
    assert browser.sent == [{"type": "session.error", "message": "llm down"}]


def test_generate_reply_records_spoken_partial_on_cancel(monkeypatch):
    # Barge-in after the first sentence is spoken: that sentence must be recorded in
    # history so the conversation keeps user/assistant alternation, even though the
    # reply never reached its normal completion.
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.SYSTEM_PROMPT = "be brief"

    async def llm_stream(_messages):
        yield "First sentence. "  # one complete sentence -> spoken
        await asyncio.Event().wait()  # then block until the barge-in cancels us

    async def connect_tts():
        return FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Audio", "is_final": True}])

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=connect_tts,
        llm_stream=llm_stream,
        settings=settings,
    )
    browser = FakeBrowser()
    session = cascade.Session()

    async def drive():
        task = asyncio.create_task(cascade._generate_reply(browser, deps, session))
        for _ in range(5):
            await asyncio.sleep(0)  # let it stream + synthesize the sentence, then block
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(task)

    asyncio.run(drive())
    assert session.history == [{"role": "assistant", "content": "First sentence."}]
