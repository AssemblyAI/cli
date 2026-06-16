"""Shared loaders and fakes for the agent-framework template tests."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pytest

TEMPLATE_DIR = Path("aai_cli/init/templates/agent_framework")


def _load(module: str, monkeypatch: pytest.MonkeyPatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in ("api.index", "api.cascade", "api.settings", "api"):
        sys.modules.pop(name, None)
    monkeypatch.syspath_prepend(str(TEMPLATE_DIR))
    return importlib.import_module(module)


def _cascade(monkeypatch: pytest.MonkeyPatch):
    return _load("api.cascade", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")


def reimport(name: str):
    """Re-fetch an already-loaded template module as an untyped handle.

    Tests mutate `settings.API_KEY = …` etc.; a `ModuleType` return would reject
    those attribute writes, so the module is laundered through its own dynamically
    typed `__dict__` to recover the open attribute handle the fakes need.
    """
    module = importlib.import_module(name)
    return module.__dict__.get("__aai_self__", module)


def untyped_bag():
    """A fresh empty dict used as a dynamic capture bag in the adapter tests.

    `json.loads` has a dynamic return type, so the bag accepts the mixed scalar /
    nested-dict values the fakes record without an explicit annotation.
    """
    return json.loads("{}")


class FakeBrowser:
    """A browser side: hands out queued inbound messages, then blocks forever so the
    mic pump stays alive until the test cancels it (mirrors a still-connected client)."""

    def __init__(self, inbound: list[dict[str, object] | None] | None = None):
        self._inbound: list[dict[str, object] | None] = list(inbound or [])
        self.sent: list[dict[str, object]] = []
        self._idle = asyncio.Event()  # never set -> recv() blocks after the queue drains

    async def send(self, event: dict[str, object]) -> None:
        self.sent.append(event)

    async def recv(self) -> dict[str, object] | None:
        if self._inbound:
            return self._inbound.pop(0)
        await self._idle.wait()
        return None

    def types(self) -> list[str]:
        return [str(event["type"]) for event in self.sent]


class FakeWS:
    """A fake STT/TTS socket: yields the given frames as JSON strings, records sends."""

    def __init__(self, frames: list[dict[str, object]] | None = None):
        self._frames: list[str] = [json.dumps(f) for f in (frames or [])]
        self.sent: list[str | bytes] = []
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

    async def send(self, data: str | bytes) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


def _async_return(value):
    async def factory():
        return value

    return factory


def _deps(monkeypatch, *, stt, tts_frames, llm_text, captured_messages=None):
    """Build a cascade + Deps wired to fakes.

    ``connect_tts`` hands out a FRESH FakeWS (cloned from ``tts_frames``) on every
    call, because a streamed reply opens one TTS socket per sentence. When
    ``captured_messages`` is a list, the fake ``llm_stream`` records the ``messages``
    it was called with into it so memory threading can be asserted.
    """
    cascade = _cascade(monkeypatch)
    settings = reimport("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    settings.GREETING = "hello!"
    settings.SYSTEM_PROMPT = "be brief"

    async def llm_stream(messages):
        if captured_messages is not None:
            captured_messages.append(messages)
        for piece in llm_text:
            yield piece

    async def connect_tts():
        return FakeWS(tts_frames)

    deps = cascade.Deps(
        connect_stt=_async_return(stt),
        connect_tts=connect_tts,
        llm_stream=llm_stream,
        settings=settings,
    )
    return cascade, deps


class _LLMChunk:
    """Mimics one OpenAI streaming chunk: `chunk.choices[0].delta.content`."""

    def __init__(self, content: str | None):
        self.choices = [type("Choice", (), {"delta": type("Delta", (), {"content": content})()})()]


class _FakeLLMStream:
    """An async-iterable over `_LLMChunk`s, the shape `client.chat.completions.create` returns."""

    def __init__(self, contents: list[str | None]):
        self._contents = contents

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for content in self._contents:
            yield _LLMChunk(content)


def _fake_openai_client(captured, contents: list[str | None]):
    """A fake `AsyncOpenAI` class recording its kwargs and the create() kwargs into `captured`."""

    class _FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeLLMStream(contents)

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.chat = type("Chat", (), {"completions": _FakeCompletions()})()

    return _FakeClient
