"""Hermetic tests for the agent-framework (cascaded voice agent) template.

The template ships a standalone FastAPI app under api/; load it by path with its
own `api` package, evicting any other template's cached `api` modules so imports
stay collision-free under pytest-xdist / pytest-randomly.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TEMPLATE_DIR = Path("aai_cli/init/templates/agent-framework")


def _load(module: str, monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in ("api.index", "api.cascade", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))
    return importlib.import_module(module)


def test_settings_imports_without_key_or_tts_host(monkeypatch):
    # Pre-set vars to empty strings so load_dotenv() (override=False by default) won't
    # overwrite them from any ambient .env found up the directory tree — the module must
    # still import cleanly (the empty-host guard lives in the WS handler, not at import).
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "")
    monkeypatch.setenv("ASSEMBLYAI_TTS_HOST", "")
    settings = _load("api.settings", monkeypatch)
    assert settings.API_KEY == ""
    assert settings.MODEL == "claude-haiku-4-5-20251001"
    assert settings.VOICE == "ivy"
    assert settings.INPUT_SAMPLE_RATE == 16000
    assert settings.OUTPUT_SAMPLE_RATE == 24000


def test_settings_reads_env(monkeypatch):
    settings = _load(
        "api.settings",
        monkeypatch,
        ASSEMBLYAI_API_KEY="sk-test",
        ASSEMBLYAI_STREAMING_HOST="streaming.example",
        ASSEMBLYAI_TTS_HOST="tts.example",
        ASSEMBLYAI_LLM_GATEWAY_URL="https://llm.example/v1",
    )
    assert settings.API_KEY == "sk-test"
    assert settings.STREAMING_HOST == "streaming.example"
    assert settings.TTS_HOST == "tts.example"
    assert settings.LLM_GATEWAY_URL == "https://llm.example/v1"


def test_settings_sandbox_defaults(monkeypatch):
    # With the host env vars unset, the module falls back to the sandbox defaults
    # (TTS is sandbox-only, so the whole cascade defaults there). Asserting the exact
    # default strings keeps the mutation gate honest on settings.py's literals.
    monkeypatch.delenv("ASSEMBLYAI_STREAMING_HOST", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_TTS_HOST", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_LLM_GATEWAY_URL", raising=False)
    settings = _load("api.settings", monkeypatch)
    assert settings.STREAMING_HOST == "streaming.sandbox000.assemblyai-labs.com"
    assert settings.TTS_HOST == "streaming-tts.sandbox000.assemblyai-labs.com"
    assert settings.LLM_GATEWAY_URL == "https://llm-gateway.sandbox000.assemblyai-labs.com/v1"
    assert settings.SYSTEM_PROMPT == (
        "You are a friendly, concise voice assistant. Keep replies short and conversational."
    )
    assert settings.GREETING == "Hi! I'm your AssemblyAI voice agent. What can I help you with?"


def _cascade(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load("api.cascade", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")


def test_unavailable_reason_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = ""
    settings.TTS_HOST = "tts.example"
    assert "ASSEMBLYAI_API_KEY" in cascade.unavailable_reason(settings)


def test_unavailable_reason_missing_tts_host(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = ""
    reason = cascade.unavailable_reason(settings)
    assert "sandbox" in reason and "assembly --sandbox init agent-framework" in reason


def test_unavailable_reason_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    assert cascade.unavailable_reason(settings) is None


def test_stt_url_carries_streaming_params(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.STREAMING_HOST = "streaming.example"
    settings.INPUT_SAMPLE_RATE = 16000
    url = cascade.stt_url(settings)
    assert url.startswith("wss://streaming.example/v3/ws?")
    assert "sample_rate=16000" in url
    assert "encoding=pcm_s16le" in url
    assert "speech_model=u3-rt-pro" in url
    assert "format_turns=true" in url


def test_tts_url_carries_voice_and_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.TTS_HOST = "tts.example"
    settings.VOICE = "ivy"
    settings.OUTPUT_SAMPLE_RATE = 24000
    url = cascade.tts_url(settings)
    assert url.startswith("wss://tts.example/v1/ws/?")
    assert "voice=ivy" in url
    assert "sample_rate=24000" in url


def test_is_final_user_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    assert cascade.is_final_user_turn({"end_of_turn": True, "turn_is_formatted": True}) is True
    assert cascade.is_final_user_turn({"end_of_turn": True, "turn_is_formatted": False}) is False
    assert cascade.is_final_user_turn({"end_of_turn": False, "turn_is_formatted": True}) is False
    assert cascade.is_final_user_turn({}) is False


def test_build_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    messages = cascade.build_messages("be brief", "hello there")
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello there"},
    ]


class FakeBrowser:
    """A browser side: hands out queued inbound messages, then blocks forever so the
    mic pump stays alive until the test cancels it (mirrors a still-connected client)."""

    def __init__(self, inbound: list[dict] | None = None):
        self._inbound = list(inbound or [])
        self.sent: list[dict] = []
        self._idle = asyncio.Event()  # never set -> recv() blocks after the queue drains

    async def send(self, event: dict) -> None:
        self.sent.append(event)

    async def recv(self) -> dict | None:
        if self._inbound:
            return self._inbound.pop(0)
        await self._idle.wait()
        return None

    def types(self) -> list[str]:
        return [event["type"] for event in self.sent]


class FakeWS:
    """A fake STT/TTS socket: yields the given frames as JSON strings, records sends."""

    def __init__(self, frames: list[dict] | None = None):
        self._frames = [json.dumps(f) for f in (frames or [])]
        self.sent: list[Any] = []
        self.closed = False

    def __aiter__(self) -> FakeWS:
        return self

    async def __anext__(self) -> str:
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def recv(self) -> str:
        if not self._frames:
            raise AssertionError("recv() past end of fake frames")
        return self._frames.pop(0)

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


def _async_return(value):
    async def factory():
        return value

    return factory


def _deps(monkeypatch, *, stt, tts_frames, llm_text):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.GREETING = "hello!"
    settings.SYSTEM_PROMPT = "be brief"

    async def llm_stream(_messages):
        for piece in llm_text:
            yield piece

    deps = cascade.Deps(
        connect_stt=_async_return(stt),
        connect_tts=_async_return(FakeWS(tts_frames)),
        llm_stream=llm_stream,
        settings=settings,
    )
    return cascade, deps


def test_pump_mic_forwards_decoded_audio(monkeypatch):
    cascade = _cascade(monkeypatch)
    pcm = b"\x01\x02\x03\x04"
    browser = FakeBrowser([{"type": "input.audio", "audio": base64.b64encode(pcm).decode()}, None])
    stt = FakeWS()
    asyncio.run(cascade._pump_mic(browser, stt))
    assert stt.sent == [pcm]


def test_pump_mic_ignores_non_audio_and_stops_on_disconnect(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser([{"type": "noise"}, None])
    stt = FakeWS()
    asyncio.run(cascade._pump_mic(browser, stt))
    assert stt.sent == []


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
    assert kinds == ["Generate", "ForceFlushTextBuffer", "Terminate"]
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


def test_generate_reply_speaks_llm_text(monkeypatch):
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {}},
            {"type": "Audio", "audio": "AAA=", "is_final": True},
        ],
        llm_text=["Hello", " world"],
    )
    browser = FakeBrowser()
    asyncio.run(cascade._generate_reply(browser, deps, cascade.build_messages("be brief", "hi")))
    assert {"type": "transcript.agent", "text": "Hello world"} in browser.sent
    assert {"type": "reply.audio", "data": "AAA="} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}


def test_generate_reply_empty_llm_emits_done(monkeypatch):
    cascade, deps = _deps(monkeypatch, stt=FakeWS(), tts_frames=[], llm_text=["  "])
    browser = FakeBrowser()
    asyncio.run(cascade._generate_reply(browser, deps, []))
    assert browser.sent == [{"type": "reply.done", "status": "empty"}]


def test_generate_reply_surfaces_error(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"

    async def llm_stream(_messages):
        raise RuntimeError("llm down")
        yield  # pragma: no cover - makes this an async generator

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=_async_return(FakeWS()),
        llm_stream=llm_stream,
        settings=settings,
    )
    browser = FakeBrowser()
    asyncio.run(cascade._generate_reply(browser, deps, []))
    assert browser.sent == [{"type": "session.error", "message": "llm down"}]


def test_maybe_barge_in_cancels_active_reply(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()

    async def drive():
        session = cascade.Session()
        started = asyncio.Event()

        async def never_ending():
            started.set()
            await asyncio.Event().wait()

        session.reply_task = asyncio.create_task(never_ending())
        await started.wait()
        await cascade.maybe_barge_in(browser, session)
        return session

    session = asyncio.run(drive())
    assert browser.sent == [{"type": "input.speech.started"}]
    assert session.reply_task is None


def test_maybe_barge_in_noop_without_reply(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    asyncio.run(cascade.maybe_barge_in(browser, cascade.Session()))
    assert browser.sent == []


def test_run_session_unavailable_emits_error(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = ""
    browser = FakeBrowser()
    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=_async_return(FakeWS()),
        llm_stream=lambda _m: iter(()),
        settings=settings,
    )
    asyncio.run(cascade.run_session(browser, deps))
    assert browser.types() == ["session.error"]


def test_run_session_connect_failure_emits_error(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"

    async def boom():
        raise RuntimeError("no route to host")

    deps = cascade.Deps(
        connect_stt=boom,
        connect_tts=_async_return(FakeWS()),
        llm_stream=lambda _m: iter(()),
        settings=settings,
    )
    browser = FakeBrowser()
    asyncio.run(cascade.run_session(browser, deps))
    assert browser.types() == ["session.error"]
    assert "no route to host" in browser.sent[0]["message"]


def test_run_session_happy_path(monkeypatch):
    # STT yields one finalized user turn, then closes -> the reply drains, then the
    # session tears down. The greeting speaks first. The mic pump blocks on FakeBrowser's
    # idle event until run_session cancels it.
    stt = FakeWS(
        [
            {
                "type": "Turn",
                "transcript": "what time is it",
                "end_of_turn": True,
                "turn_is_formatted": True,
            }
        ]
    )
    tts_sockets = [
        FakeWS(
            [
                {"type": "Begin", "configuration": {}},
                {"type": "Audio", "audio": "G=", "is_final": True},
            ]
        ),
        FakeWS(
            [
                {"type": "Begin", "configuration": {}},
                {"type": "Audio", "audio": "R=", "is_final": True},
            ]
        ),
    ]
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.GREETING = "hello!"
    settings.SYSTEM_PROMPT = "be brief"

    async def llm_stream(_messages):
        yield "It is noon."

    def connect_tts():
        async def factory():
            return tts_sockets.pop(0)

        return factory()

    deps = cascade.Deps(
        connect_stt=_async_return(stt),
        connect_tts=connect_tts,
        llm_stream=llm_stream,
        settings=settings,
    )
    browser = FakeBrowser()
    asyncio.run(asyncio.wait_for(cascade.run_session(browser, deps), timeout=5))

    types = browser.types()
    assert types[0] == "transcript.agent"  # greeting spoken first
    assert {"type": "transcript.user", "text": "what time is it"} in browser.sent
    assert {"type": "transcript.agent", "text": "It is noon."} in browser.sent
    assert {"type": "reply.audio", "data": "R="} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}
    assert stt.closed is True


def test_pump_stt_emits_user_transcript_and_barges_in(monkeypatch):
    # A non-final turn with text emits transcript.user and barges in on an active reply.
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    browser = FakeBrowser()

    async def drive():
        session = cascade.Session()

        async def never_ending():
            await asyncio.Event().wait()

        session.reply_task = asyncio.create_task(never_ending())
        stt = FakeWS(
            [
                {
                    "type": "Turn",
                    "transcript": "wait",
                    "end_of_turn": False,
                    "turn_is_formatted": False,
                }
            ]
        )
        # _deps not used; build minimal deps
        deps = cascade.Deps(
            connect_stt=_async_return(stt),
            connect_tts=_async_return(FakeWS()),
            llm_stream=lambda _m: iter(()),
            settings=settings,
        )
        await cascade._pump_stt(browser, stt, deps, session)

    asyncio.run(asyncio.wait_for(drive(), timeout=5))
    assert {"type": "transcript.user", "text": "wait"} in browser.sent
    assert {"type": "input.speech.started"} in browser.sent
