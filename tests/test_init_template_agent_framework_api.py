"""Hermetic tests for the agent-framework template's live-client adapters and FastAPI routes."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from tests._agent_framework import (
    FakeBrowser,
    FakeWS,
    _async_return,
    _cascade,
    _fake_openai_client,
    _load,
    reimport,
    untyped_bag,
)


def test_connect_stt_uses_auth_header_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import websockets

    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.STREAMING_HOST = "streaming.example"
    settings.INPUT_SAMPLE_RATE = 16000
    captured = untyped_bag()

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeWS()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    result = asyncio.run(cascade._connect_stt(settings))
    assert isinstance(result, FakeWS)
    assert captured["url"] == cascade.stt_url(settings)
    assert captured["kwargs"]["additional_headers"] == {"Authorization": "sk-test"}


def test_connect_tts_passes_max_size_none(monkeypatch: pytest.MonkeyPatch) -> None:
    import websockets

    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.VOICE = "jane"
    settings.OUTPUT_SAMPLE_RATE = 24000
    captured = untyped_bag()

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeWS()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    result = asyncio.run(cascade._connect_tts(settings))
    assert isinstance(result, FakeWS)
    assert captured["url"] == cascade.tts_url(settings)
    assert captured["kwargs"]["additional_headers"] == {"Authorization": "sk-test"}
    assert captured["kwargs"]["max_size"] is None


def test_llm_stream_yields_nonempty_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.LLM_GATEWAY_URL = "https://llm.example/v1"
    settings.MODEL = "test-model"
    captured = untyped_bag()

    monkeypatch.setattr(
        openai, "AsyncOpenAI", _fake_openai_client(captured, ["Hi", "", " there", None])
    )

    async def collect():
        return [d async for d in cascade._llm_stream(settings, [{"role": "user", "content": "hi"}])]

    deltas = asyncio.run(collect())
    assert deltas == ["Hi", " there"]  # empty + None filtered by `if delta`
    assert captured["model"] == "test-model"
    assert captured["stream"] is True
    assert captured["client_kwargs"]["base_url"] == "https://llm.example/v1"
    assert captured["client_kwargs"]["api_key"] == "sk-test"


def test_llm_stream_skips_chunk_without_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Anthropic-backed gateway ends the stream with a usage/final chunk carrying an
    # empty `choices` list; _llm_stream must skip it, not IndexError on chunk.choices[0].
    import openai

    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.LLM_GATEWAY_URL = "https://llm.example/v1"
    settings.MODEL = "test-model"

    def _chunk(content: str) -> SimpleNamespace:
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

    class _Stream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            yield _chunk("Hi")
            yield SimpleNamespace(choices=[])  # gateway usage/final chunk — no choices
            yield _chunk(" there")

    class _Completions:
        async def create(self, **kwargs):
            return _Stream()

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)

    async def collect():
        return [d async for d in cascade._llm_stream(settings, [{"role": "user", "content": "hi"}])]

    assert asyncio.run(collect()) == ["Hi", " there"]


def test_deps_real_factories_invoke_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    import websockets

    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.STREAMING_HOST = "streaming.example"
    settings.LLM_GATEWAY_URL = "https://llm.example/v1"

    async def fake_connect(url, **kwargs):
        return FakeWS()

    monkeypatch.setattr(websockets, "connect", fake_connect)
    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_client({}, []))

    deps = cascade.Deps.real(settings)
    assert deps.settings is settings

    async def drive():
        assert isinstance(await deps.connect_stt(), FakeWS)
        assert isinstance(await deps.connect_tts(), FakeWS)
        return [d async for d in deps.llm_stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(drive()) == []


def test_generate_reply_propagates_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    browser = FakeBrowser()

    async def llm_stream(_messages):
        yield "partial"  # one delta, then block until the task is cancelled
        await asyncio.Event().wait()

    deps = cascade.Deps(
        connect_stt=_async_return(FakeWS()),
        connect_tts=_async_return(FakeWS()),
        llm_stream=llm_stream,
        settings=settings,
    )

    async def drive():
        task = asyncio.create_task(cascade._generate_reply(browser, deps, cascade.Session()))
        await asyncio.sleep(0)  # let it start and block on the LLM
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.gather(task)

    asyncio.run(drive())
    # Cancellation must NOT be turned into a session.error.
    assert browser.sent == []


def test_pump_stt_skips_non_turn_and_empty_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.SYSTEM_PROMPT = "be brief"
    browser = FakeBrowser()
    # A non-Turn frame and an empty-transcript Turn must both be skipped (no transcript.user),
    # then the stream closes.
    stt = FakeWS(
        [
            {"type": "Begin"},
            {"type": "Turn", "transcript": "", "end_of_turn": False, "turn_is_formatted": False},
        ]
    )
    deps = cascade.Deps(
        connect_stt=_async_return(stt),
        connect_tts=_async_return(FakeWS()),
        llm_stream=lambda _m: iter(()),
        settings=settings,
    )
    session = cascade.Session()
    asyncio.run(asyncio.wait_for(cascade._pump_stt(browser, stt, deps, session), timeout=5))
    assert browser.sent == []  # nothing emitted for non-Turn or empty-transcript frames


def test_synthesize_audio_frame_missing_payload_defaults_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    # An Audio frame with no `audio` key must yield reply.audio with data == "" (the default).
    tts = FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Audio", "is_final": True}])
    asyncio.run(cascade._synthesize(browser, tts, "hi"))
    assert {"type": "reply.audio", "data": ""} in browser.sent


def test_index_serves_page(monkeypatch: pytest.MonkeyPatch) -> None:
    index = _load("api.index", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")
    from fastapi.testclient import TestClient

    resp = TestClient(index.app).get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_ws_route_runs_cascade(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drive the real /ws adapter with TestClient's WebSocket, but stub run_session so
    # the route's accept + adapter wiring is exercised without real upstreams.
    index = _load("api.index", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")
    cascade = reimport("api.cascade")
    captured = untyped_bag()

    async def fake_run_session(browser, deps):
        captured["deps"] = deps
        msg = await browser.recv()
        await browser.send({"type": "echo", "got": msg})

    monkeypatch.setattr(cascade, "run_session", fake_run_session)
    from fastapi.testclient import TestClient

    with TestClient(index.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "input.audio", "audio": "AAA="})
        assert ws.receive_json() == {
            "type": "echo",
            "got": {"type": "input.audio", "audio": "AAA="},
        }
    # The handler must wire the real settings module into Deps.real (not None / wrong arg).
    assert captured["deps"].settings is importlib.import_module("api.settings")


def test_fastapi_browser_recv_returns_none_on_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    cascade = _cascade(monkeypatch)
    from fastapi import WebSocketDisconnect

    class FakeWSStarlette:
        def __init__(self):
            self.sent: list[dict[str, object]] = []

        async def send_json(self, event):
            self.sent.append(event)

        async def receive_json(self):
            raise WebSocketDisconnect(code=1000)

    ws = FakeWSStarlette()
    browser = cascade.FastAPIBrowser(ws)

    async def drive():
        await browser.send({"type": "x"})
        return await browser.recv()

    assert asyncio.run(drive()) is None
    assert ws.sent == [{"type": "x"}]
