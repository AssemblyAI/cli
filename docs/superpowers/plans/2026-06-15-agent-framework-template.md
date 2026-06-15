# `agent-framework` init template — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth `assembly init` starter template, `agent-framework`, with the same browser UI as `voice-agent` but built on a server-orchestrated **cascade** — Streaming STT → LLM Gateway → sandbox TTS — instead of the all-in-one Voice Agent endpoint.

**Architecture:** The browser opens one same-origin WebSocket (`/ws`) to a FastAPI backend. The backend runs the cascade: forwards mic PCM to the Streaming v3 STT socket, detects end-of-turn, streams the transcript through the OpenAI-compatible LLM Gateway, synthesizes the reply over the sandbox streaming-TTS socket, and streams audio back. All three API credentials stay server-side. The orchestrator (`api/cascade.py`) is built with injected connect-factories + LLM callable so it is fully testable with fakes (mirroring `aai_cli/tts/session.py`).

**Tech Stack:** FastAPI + Starlette WebSockets, `websockets` async client (STT + TTS), `openai.AsyncOpenAI` (streamed gateway completion), `uvicorn`. Buildless static HTML/CSS/JS frontend.

---

## Key constraints discovered (read before starting)

- **Sandbox-only.** `streaming_tts_host` is empty in production, so the whole cascade must target `sandbox000` with a sandbox key. The backend fails fast (a `session.error` event, *not* an import error) when the TTS host is empty.
- **Settings must not raise at import.** `tests/test_init_template_serve.py::test_serves_root_and_static_assets` is parametrized over every template dir; it imports `api.index` and hits `GET /`. The empty-TTS-host guard therefore lives in the WS handler, never at module import.
- **Template `.py` is coverage- and mutation-gated.** Confirmed: `coverage.xml` includes `init/templates/.../api/*.py`. diff-cover requires 100% patch coverage of new template lines and the mutation gate mutates them, so `cascade.py` needs real, asserting tests.
- **The contract `_STDLIB` set is incomplete.** `tests/test_init_template_contract.py::test_requirements_cover_backend_imports` treats any import not in its `_STDLIB` set as third-party and demands it in `requirements.txt`. `asyncio`, `base64`, `contextlib`, `dataclasses`, `collections`, `urllib` are stdlib we use — extend `_STDLIB` (Task 1).
- **Two hard-coded test assertions break when the registry grows:** `tests/test_init_command.py::test_init_template_arg_help_is_derived_from_registry` (exact help string) and the `assembly init --help` snapshot `tests/__snapshots__/test_snapshots_help_build.ambr`. Update the first by hand (Task 1); regenerate the second with `--snapshot-update` (Task 9).
- `openai>=2.41.0` and `websockets>=16.0` are **main project deps**, so the dev env can import `cascade.py` for the serve test. The template's own `requirements.txt` pins its independent floors.

## File structure

New template dir `aai_cli/init/templates/agent-framework/`:
- `api/__init__.py` — empty package marker.
- `api/settings.py` — env-derived config; no import-time raise.
- `api/cascade.py` — pure helpers + the injectable async orchestrator + the FastAPI browser adapter.
- `api/index.py` — FastAPI app: static mount, `GET /`, `@app.websocket("/ws")` adapter.
- `static/index.html` — copy of voice-agent's page, cascade-worded.
- `static/styles.css` — verbatim copy of voice-agent's.
- `static/audio.js` — verbatim copy of voice-agent's.
- `static/app.js` — same event handling as voice-agent; connects to `/ws` directly.
- `README.md`, `AGENTS.md`, `env.example`, `gitignore`, `requirements.txt`, `Procfile`, `Dockerfile`, `dockerignore`, `runtime.txt`, `vercel.json`.

Shared CLI edits:
- `aai_cli/init/templates.py` — register the template.
- `aai_cli/app/init_exec.py` — inject `ASSEMBLYAI_TTS_HOST` into scaffolded `.env`.

Test edits:
- `tests/test_init_template_contract.py` — extend `_STDLIB`.
- `tests/test_init_command.py` — update the exact help-string assertion.
- `tests/test_init_template_agent_framework.py` — NEW bespoke tests.
- `tests/__snapshots__/test_snapshots_help_build.ambr` — regenerated.

---

## Task 1: CLI wiring (register template, inject TTS host, fix gated assertions)

**Files:**
- Modify: `aai_cli/init/templates.py`
- Modify: `aai_cli/app/init_exec.py:91-104`
- Modify: `tests/test_init_template_contract.py` (the `_STDLIB` constant)
- Modify: `tests/test_init_command.py` (exact help string)
- Test: `tests/test_init_command.py`, `tests/test_init_templates.py`

- [ ] **Step 1: Update the failing registry expectations first (TDD red)**

In `tests/test_init_command.py`, update the exact-help assertion to include the new id (appended last):

```python
    assert default.help == (
        "Template to scaffold: audio-transcription, live-captions, voice-agent, "
        "agent-framework (omit to pick interactively)"
    )
```

- [ ] **Step 2: Run it to confirm it now fails (registry not updated yet)**

Run: `uv run pytest tests/test_init_command.py::test_init_template_arg_help_is_derived_from_registry tests/test_init_templates.py -q`
Expected: FAIL — `test_order_matches_registry`/`test_every_shipped_directory_is_registered` and the help-string test disagree with the registry.

- [ ] **Step 3: Register the template**

In `aai_cli/init/templates.py`, add the entry and order (append after `voice-agent`):

```python
TEMPLATES: dict[str, str] = {
    "audio-transcription": "Audio Transcription",
    "live-captions": "Live Captions",
    "voice-agent": "Voice Agent",
    "agent-framework": "Agent Framework",
}

# Display order for the picker and `--help`.
TEMPLATE_ORDER: tuple[str, ...] = (
    "audio-transcription",
    "live-captions",
    "voice-agent",
    "agent-framework",
)
```

- [ ] **Step 4: Inject the TTS host into scaffolded `.env`**

In `aai_cli/app/init_exec.py`, add the TTS host to `_active_env_vars()` (the cascade template reads it; empty in prod, which the template treats as "sandbox required"):

```python
    return {
        "ASSEMBLYAI_BASE_URL": env.api_base,
        "ASSEMBLYAI_LLM_GATEWAY_URL": env.llm_gateway_base,
        "ASSEMBLYAI_STREAMING_HOST": env.streaming_host,
        # Voice Agent host mirrors the streaming host's naming across environments.
        "ASSEMBLYAI_AGENTS_HOST": env.streaming_host.replace("streaming", "agents", 1),
        # Streaming-TTS host for the cascade (agent-framework) template. Empty in
        # production, where streaming TTS has no host; that template then refuses to
        # run and points at --sandbox.
        "ASSEMBLYAI_TTS_HOST": env.streaming_tts_host,
    }
```

- [ ] **Step 5: Extend the contract test's stdlib set**

In `tests/test_init_template_contract.py`, widen `_STDLIB` so the cascade's stdlib imports aren't mistaken for third-party packages:

```python
_STDLIB = {
    "os",
    "tempfile",
    "uuid",
    "pathlib",
    "__future__",
    "json",
    "typing",
    "asyncio",
    "base64",
    "contextlib",
    "dataclasses",
    "collections",
    "urllib",
}
```

- [ ] **Step 6: Add an assertion pinning the new env var (mutation coverage for Step 4)**

In `tests/test_init_command.py`, beside the existing `_active_env_vars` test (~line 312), add:

```python
def test_active_env_vars_includes_streaming_tts_host(monkeypatch):
    fake = SimpleNamespace(
        api_base="https://api.x",
        llm_gateway_base="https://llm.x/v1",
        streaming_host="streaming.x",
        streaming_tts_host="streaming-tts.x",
    )
    monkeypatch.setattr(init_exec.environments, "active", lambda: fake)
    assert init_exec._active_env_vars()["ASSEMBLYAI_TTS_HOST"] == "streaming-tts.x"
```

(Use the same `SimpleNamespace`/`monkeypatch` shape as the neighboring test; import `SimpleNamespace` from `types` if not already imported.)

- [ ] **Step 7: Run the registry + command tests (they pass except for the missing dir)**

Run: `uv run pytest tests/test_init_templates.py tests/test_init_command.py -q`
Expected: `test_every_registered_template_has_a_directory` FAILS (dir not created yet); everything else PASSES. This failure is resolved in Task 6 when `api/index.py` lands. Proceed.

- [ ] **Step 8: Commit**

```bash
git add aai_cli/init/templates.py aai_cli/app/init_exec.py tests/test_init_template_contract.py tests/test_init_command.py
git commit -m "feat(init): register agent-framework template + inject TTS host"
```

---

## Task 2: Template skeleton + verbatim static assets

**Files:**
- Create: `aai_cli/init/templates/agent-framework/api/__init__.py`
- Create (copy): `aai_cli/init/templates/agent-framework/static/styles.css`
- Create (copy): `aai_cli/init/templates/agent-framework/static/audio.js`

- [ ] **Step 1: Create the directory and copy the verbatim assets**

Run:

```bash
SRC=aai_cli/init/templates/voice-agent
DST=aai_cli/init/templates/agent-framework
mkdir -p "$DST/api" "$DST/static"
: > "$DST/api/__init__.py"
cp "$SRC/static/styles.css" "$DST/static/styles.css"
cp "$SRC/static/audio.js" "$DST/static/audio.js"
```

`styles.css` and `audio.js` are reused unchanged — the UI and the mic-pipeline/PCM-player/barge-in helpers are identical to `voice-agent`.

- [ ] **Step 2: Verify the copies are byte-identical**

Run: `diff aai_cli/init/templates/voice-agent/static/styles.css aai_cli/init/templates/agent-framework/static/styles.css && diff aai_cli/init/templates/voice-agent/static/audio.js aai_cli/init/templates/agent-framework/static/audio.js && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add aai_cli/init/templates/agent-framework/api/__init__.py aai_cli/init/templates/agent-framework/static/styles.css aai_cli/init/templates/agent-framework/static/audio.js
git commit -m "feat(agent-framework): skeleton + shared static assets"
```

---

## Task 3: `settings.py` + availability guard

**Files:**
- Create: `aai_cli/init/templates/agent-framework/api/settings.py`
- Test: `tests/test_init_template_agent_framework.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_init_template_agent_framework.py`:

```python
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
    # isolate_env strips ambient vars; with nothing set the module must still import
    # (the empty-host guard lives in the WS handler, not at import).
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_TTS_HOST", raising=False)
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.settings'`.

- [ ] **Step 3: Write `settings.py`**

Create `aai_cli/init/templates/agent-framework/api/settings.py`:

```python
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

# Hosts. `assembly init` pins these to the active environment. Streaming TTS only
# exists in the sandbox, so this whole cascade is sandbox-only (see README); the
# defaults point at the sandbox so a bare clone works with a sandbox key.
STREAMING_HOST = os.environ.get(
    "ASSEMBLYAI_STREAMING_HOST", "streaming.sandbox000.assemblyai-labs.com"
)
TTS_HOST = os.environ.get("ASSEMBLYAI_TTS_HOST", "streaming-tts.sandbox000.assemblyai-labs.com")
LLM_GATEWAY_URL = os.environ.get(
    "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.sandbox000.assemblyai-labs.com/v1"
)

# The cascade's three knobs — edit these to change behavior.
MODEL = "claude-haiku-4-5-20251001"
VOICE = "ivy"
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Keep replies short and conversational."
)
GREETING = "Hi! I'm your AssemblyAI voice agent. What can I help you with?"

# 16 kHz PCM in (Streaming v3); 24 kHz PCM out (streaming TTS).
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/templates/agent-framework/api/settings.py tests/test_init_template_agent_framework.py
git commit -m "feat(agent-framework): settings module"
```

---

## Task 4: `cascade.py` pure helpers

**Files:**
- Create: `aai_cli/init/templates/agent-framework/api/cascade.py`
- Test: `tests/test_init_template_agent_framework.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_init_template_agent_framework.py`:

```python
def _cascade(monkeypatch) -> ModuleType:
    return _load("api.cascade", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")


def test_unavailable_reason_missing_key(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = ""
    settings.TTS_HOST = "tts.example"
    assert "ASSEMBLYAI_API_KEY" in cascade.unavailable_reason(settings)


def test_unavailable_reason_missing_tts_host(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = ""
    reason = cascade.unavailable_reason(settings)
    assert "sandbox" in reason and "assembly --sandbox init agent-framework" in reason


def test_unavailable_reason_ok(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.API_KEY = "sk-test"
    settings.TTS_HOST = "tts.example"
    assert cascade.unavailable_reason(settings) is None


def test_stt_url_carries_streaming_params(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.STREAMING_HOST = "streaming.example"
    settings.INPUT_SAMPLE_RATE = 16000
    url = cascade.stt_url(settings)
    assert url.startswith("wss://streaming.example/v3/ws?")
    assert "sample_rate=16000" in url
    assert "encoding=pcm_s16le" in url
    assert "format_turns=true" in url


def test_tts_url_carries_voice_and_rate(monkeypatch):
    cascade = _cascade(monkeypatch)
    settings = importlib.import_module("api.settings")
    settings.TTS_HOST = "tts.example"
    settings.VOICE = "ivy"
    settings.OUTPUT_SAMPLE_RATE = 24000
    url = cascade.tts_url(settings)
    assert url.startswith("wss://tts.example/v1/ws/?")
    assert "voice=ivy" in url
    assert "sample_rate=24000" in url


def test_is_final_user_turn(monkeypatch):
    cascade = _cascade(monkeypatch)
    assert cascade.is_final_user_turn({"end_of_turn": True, "turn_is_formatted": True}) is True
    assert cascade.is_final_user_turn({"end_of_turn": True, "turn_is_formatted": False}) is False
    assert cascade.is_final_user_turn({"end_of_turn": False, "turn_is_formatted": True}) is False
    assert cascade.is_final_user_turn({}) is False


def test_build_messages(monkeypatch):
    cascade = _cascade(monkeypatch)
    messages = cascade.build_messages("be brief", "hello there")
    assert messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello there"},
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q -k "unavailable or url or final or build"`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.cascade'`.

- [ ] **Step 3: Write the helper section of `cascade.py`**

Create `aai_cli/init/templates/agent-framework/api/cascade.py` with the imports and pure helpers (the orchestrator is added in Task 5, the adapter in Task 6 — write them as one growing file):

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q -k "unavailable or url or final or build"`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/templates/agent-framework/api/cascade.py tests/test_init_template_agent_framework.py
git commit -m "feat(agent-framework): cascade pure helpers"
```

---

## Task 5: `cascade.py` orchestrator (the cascade itself)

**Files:**
- Modify: `aai_cli/init/templates/agent-framework/api/cascade.py` (append)
- Test: `tests/test_init_template_agent_framework.py` (append)

- [ ] **Step 1: Write the failing tests (fakes + each stage + happy path)**

Append to `tests/test_init_template_agent_framework.py`:

```python
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

    def __aiter__(self) -> "FakeWS":
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


def _async_return(value):
    async def factory():
        return value

    return factory


def test_pump_mic_forwards_decoded_audio(monkeypatch):
    cascade = _cascade(monkeypatch)
    pcm = b"\x01\x02\x03\x04"
    browser = FakeBrowser([{"type": "input.audio", "audio": base64.b64encode(pcm).decode()}])
    stt = FakeWS()

    async def drive():
        # recv() returns the one message, then we cancel by feeding a disconnect.
        browser._inbound.append(None)  # type: ignore[arg-type]
        await cascade._pump_mic(browser, stt)

    asyncio.run(drive())
    assert stt.sent == [pcm]


def test_pump_mic_ignores_non_audio_and_stops_on_disconnect(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser([{"type": "noise"}, None])  # type: ignore[list-item]
    stt = FakeWS()
    asyncio.run(cascade._pump_mic(browser, stt))
    assert stt.sent == []


def test_synthesize_streams_audio_frames(monkeypatch):
    cascade, deps = _deps(
        monkeypatch,
        stt=FakeWS(),
        tts_frames=[
            {"type": "Begin", "configuration": {"sample_rate": 24000}},
            {"type": "Audio", "audio": "AAA="},
            {"type": "Audio", "audio": "BBB=", "is_final": True},
        ],
        llm_text=[],
    )
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
    # Generate + ForceFlushTextBuffer + Terminate were sent.
    kinds = [json.loads(s)["type"] for s in tts.sent]
    assert kinds == ["Generate", "ForceFlushTextBuffer", "Terminate"]
    assert tts.closed is True


def test_synthesize_raises_on_error_frame(monkeypatch):
    cascade = _cascade(monkeypatch)
    browser = FakeBrowser()
    tts = FakeWS(
        [{"type": "Begin", "configuration": {}}, {"type": "Error", "error": "bad voice"}]
    )
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


def test_run_session_happy_path(monkeypatch):
    # STT yields one finalized user turn, then closes -> the reply drains, then the
    # session tears down. The greeting speaks first.
    stt = FakeWS(
        [{"type": "Turn", "transcript": "what time is it", "end_of_turn": True, "turn_is_formatted": True}]
    )

    # Each connect_tts call returns a fresh socket (greeting + reply).
    tts_sockets = [
        FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Audio", "audio": "G=", "is_final": True}]),
        FakeWS([{"type": "Begin", "configuration": {}}, {"type": "Audio", "audio": "R=", "is_final": True}]),
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
    # Greeting (agent transcript + audio + done), then the user turn, then the reply.
    assert types[0] == "transcript.agent"  # greeting text
    assert {"type": "transcript.user", "text": "what time is it"} in browser.sent
    assert {"type": "transcript.agent", "text": "It is noon."} in browser.sent
    assert {"type": "reply.audio", "data": "R="} in browser.sent
    assert browser.sent[-1] == {"type": "reply.done", "status": "completed"}
    assert stt.closed is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q -k "pump_mic or synthesize or generate_reply or barge or run_session"`
Expected: FAIL — `Deps`, `Session`, `run_session`, `_synthesize`, etc. don't exist yet.

- [ ] **Step 3: Append the orchestrator to `cascade.py`**

Add to `aai_cli/init/templates/agent-framework/api/cascade.py` (after the helpers):

```python
@dataclass
class Deps:
    """Injected cascade dependencies. `Deps.real(settings)` wires the live clients;
    tests pass fakes with the same shapes."""

    connect_stt: Callable[[], Awaitable[Any]]
    connect_tts: Callable[[], Awaitable[Any]]
    llm_stream: Callable[[list[dict[str, str]]], AsyncIterator[str]]
    settings: Any

    @classmethod
    def real(cls, settings: Any) -> "Deps":
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
    except Exception as exc:  # noqa: BLE001 — any leg failure becomes one clean event
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
    except Exception as exc:  # noqa: BLE001
        await browser.send({"type": "session.error", "message": f"Could not start the session: {exc}"})
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q -k "pump_mic or synthesize or generate_reply or barge or run_session"`
Expected: PASS. If `test_run_session_happy_path` is flaky on task scheduling, it should not be — the greeting reply is set before pumps start and `_pump_stt` drains the reply before returning, and the mic pump blocks on `FakeBrowser`'s idle event. If a hang occurs, the `asyncio.wait_for(..., timeout=5)` fails loudly rather than wedging.

- [ ] **Step 5: Format + lint the template module**

Run: `uv run ruff format aai_cli/init/templates/agent-framework/api/cascade.py && uv run ruff check aai_cli/init/templates/agent-framework/api/cascade.py`
Expected: clean (S105/TID251 are ignored for templates; the `# noqa: BLE001` keeps the broad-except lines clean).

- [ ] **Step 6: Commit**

```bash
git add aai_cli/init/templates/agent-framework/api/cascade.py tests/test_init_template_agent_framework.py
git commit -m "feat(agent-framework): cascade orchestrator"
```

---

## Task 6: `api/index.py` — FastAPI app + WebSocket adapter

**Files:**
- Create: `aai_cli/init/templates/agent-framework/api/index.py`
- Modify: `aai_cli/init/templates/agent-framework/api/cascade.py` (add `FastAPIBrowser`)
- Test: `tests/test_init_template_agent_framework.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_init_template_agent_framework.py`:

```python
def test_index_serves_page(monkeypatch):
    index = _load("api.index", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")
    from fastapi.testclient import TestClient

    resp = TestClient(index.app).get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_ws_route_runs_cascade(monkeypatch):
    # Drive the real /ws adapter with TestClient's WebSocket, but stub run_session so
    # the route's accept + adapter wiring is exercised without real upstreams.
    index = _load("api.index", monkeypatch, ASSEMBLYAI_API_KEY="sk-test")
    cascade = importlib.import_module("api.cascade")

    async def fake_run_session(browser, _deps):
        msg = await browser.recv()
        await browser.send({"type": "echo", "got": msg})

    monkeypatch.setattr(cascade, "run_session", fake_run_session)
    from fastapi.testclient import TestClient

    with TestClient(index.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "input.audio", "audio": "AAA="})
        assert ws.receive_json() == {"type": "echo", "got": {"type": "input.audio", "audio": "AAA="}}


def test_fastapi_browser_recv_returns_none_on_disconnect(monkeypatch):
    cascade = _cascade(monkeypatch)
    from fastapi import WebSocketDisconnect

    class FakeWSStarlette:
        def __init__(self):
            self.sent: list[dict] = []

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q -k "index_serves or ws_route or fastapi_browser"`
Expected: FAIL — `api.index` and `cascade.FastAPIBrowser` don't exist.

- [ ] **Step 3: Add `FastAPIBrowser` to `cascade.py`**

Append to `aai_cli/init/templates/agent-framework/api/cascade.py`:

```python
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
```

- [ ] **Step 4: Write `api/index.py`**

Create `aai_cli/init/templates/agent-framework/api/index.py`:

```python
"""Talk to a cascaded voice agent — AssemblyAI agent-framework starter (FastAPI).

The browser opens one WebSocket to this backend, which runs the cascade itself —
Streaming STT -> LLM Gateway -> streaming TTS — so your API key never reaches the
client. Streaming TTS is sandbox-only, so scaffold with `assembly --sandbox init
agent-framework` and use a sandbox key.

  WS /ws  <- {"type":"input.audio","audio":<b64 pcm>} ; -> transcripts + reply.audio
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import cascade, settings

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Accept the browser socket and run one cascade session over it."""
    await websocket.accept()
    browser = cascade.FastAPIBrowser(websocket)
    await cascade.run_session(browser, cascade.Deps.real(settings))
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_init_template_agent_framework.py -q`
Expected: PASS (all tests). Also confirm the registry directory test now passes:
Run: `uv run pytest tests/test_init_templates.py -q`
Expected: PASS.

- [ ] **Step 6: Format + lint**

Run: `uv run ruff format aai_cli/init/templates/agent-framework/api/ && uv run ruff check aai_cli/init/templates/agent-framework/api/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add aai_cli/init/templates/agent-framework/api/index.py aai_cli/init/templates/agent-framework/api/cascade.py tests/test_init_template_agent_framework.py
git commit -m "feat(agent-framework): FastAPI app + websocket adapter"
```

---

## Task 7: Frontend — `index.html` + `app.js`

**Files:**
- Create: `aai_cli/init/templates/agent-framework/static/index.html`
- Create: `aai_cli/init/templates/agent-framework/static/app.js`

- [ ] **Step 1: Write `static/index.html`**

Create `aai_cli/init/templates/agent-framework/static/index.html` (same structure/IDs/classes as voice-agent, cascade-worded copy):

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Talk to a cascaded voice agent · AssemblyAI</title>
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body class="template-page voice-agent-template">
  <main class="app-shell">
    <a class="brand" href="https://www.assemblyai.com" target="_blank" rel="noopener">
      <img class="brand-logo" src="https://www.assemblyai.com/_aai/images/logos/assemblyai-logo-primary.svg" alt="AssemblyAI" />
    </a>

    <header class="page-header">
      <span class="eyebrow">Streaming STT · LLM Gateway · TTS</span>
      <h1 class="page-title">Talk to a cascaded voice agent</h1>
      <p class="page-subtitle">Connect and just talk. This agent is a cascade your backend wires together — Streaming STT transcribes you, the LLM Gateway replies, and streaming TTS speaks it back, with turn detection and barge-in handled server-side. Your API key stays on the server. Headphones give the cleanest result.</p>
    </header>

    <div class="control-bar">
      <button id="conn" class="button connection-button" data-state="idle">● Connect</button>
      <span id="status" class="status-pill" aria-live="polite"></span>
    </div>

    <div id="log" class="conversation-log"></div>

    <footer class="page-footer">
      <span>Built with AssemblyAI</span>
      <a class="footer-link" href="https://www.assemblyai.com" target="_blank" rel="noopener">assemblyai.com →</a>
    </footer>
  </main>

  <script src="/static/audio.js"></script>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `static/app.js`**

Create `aai_cli/init/templates/agent-framework/static/app.js`. Same event handling as voice-agent's `onEvent`/`addTurn`/`bargeIn` (so `audio.js` and the UI carry over), but it opens the same-origin `/ws` directly — no token fetch, no `session.update`:

```javascript
const SESSION_CONFIG = {
  inputSampleRate: 16000,
  outputSampleRate: 24000,
  processorBufferSize: 4096,
  microphone: { audio: { echoCancellation: true, noiseSuppression: false } },
};

const connBtn = document.getElementById("conn");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

let ws = null;
let micPipeline = null;
let player = null;
let connected = false;

connBtn.addEventListener("click", () =>
  connected ? hangup() : connect().catch(fail),
);

function setStatus(message, state) {
  statusEl.textContent = message;
  statusEl.dataset.state = state;
}

function wsUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws`;
}

async function connect() {
  setStatus("Connecting...", "idle");
  ws = new WebSocket(wsUrl());
  ws.onopen = () => startMic().catch(fail);
  ws.onmessage = (event) => onEvent(JSON.parse(event.data));
  ws.onerror = () => fail("WebSocket error");
  ws.onclose = () => {
    if (connected) hangup();
  };
}

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia(
    SESSION_CONFIG.microphone,
  );
  micPipeline = AudioHelpers.createMicrophonePipeline(stream, {
    bufferSize: SESSION_CONFIG.processorBufferSize,
  });
  player = AudioHelpers.createPcmPlayer({
    sampleRate: SESSION_CONFIG.outputSampleRate,
  });
  await player.resume();
  await micPipeline.start((frame, sampleRate) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = AudioHelpers.downsampleToPCM(
      frame,
      sampleRate,
      SESSION_CONFIG.inputSampleRate,
    );
    ws.send(
      JSON.stringify({
        type: "input.audio",
        audio: AudioHelpers.bytesToB64(pcm),
      }),
    );
  });

  connected = true;
  connBtn.textContent = "■ Hang up";
  connBtn.dataset.state = "connected";
  setStatus("● Connected - just talk", "live");
}

function onEvent(event) {
  switch (event.type) {
    case "transcript.user":
      return addTurn("you", "You", event.text);
    case "transcript.agent":
      return addTurn("agent", "Agent", event.text);
    case "reply.audio":
      return player.playBase64Chunk(event.data);
    case "input.speech.started":
      return bargeIn();
    case "reply.done":
      if (event.status === "interrupted") bargeIn();
      return;
    case "session.error":
      return fail(event.message || "session error");
  }
}

function bargeIn() {
  if (player) player.stopQueuedAudio();
}

function addTurn(speakerKind, speaker, text) {
  if (!text) return;
  const turn = document.createElement("div");
  turn.className = "conversation-turn";
  turn.dataset.speaker = speakerKind;
  const who = document.createElement("span");
  who.className = "turn-speaker";
  who.textContent = speaker + ": ";
  turn.append(who, document.createTextNode(text));
  logEl.appendChild(turn);
  turn.scrollIntoView({ block: "end" });
}

function hangup() {
  connected = false;
  connBtn.textContent = "● Connect";
  connBtn.dataset.state = "idle";
  setStatus("Disconnected", "idle");
  bargeIn();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close();
  if (micPipeline) micPipeline.close();
  if (player) player.close();
  ws = null;
  micPipeline = null;
  player = null;
}

function fail(message) {
  setStatus("Error: " + message, "error");
  if (connected) hangup();
}
```

- [ ] **Step 3: Prettier-format the JS/CSS (the gate runs `prettier --check`)**

Run: `prettier --write "aai_cli/init/templates/agent-framework/static/*.js" "aai_cli/init/templates/agent-framework/static/*.css"`
Then verify: `prettier --check "aai_cli/init/templates/agent-framework/static/*.{js,css}"`
Expected: "All matched files use Prettier code style!"

- [ ] **Step 4: Verify the frontend↔backend route contract + static refs**

Run: `uv run pytest "tests/test_init_template_contract.py::test_static_assets_referenced_by_html_exist[agent-framework]" "tests/test_init_template_contract.py::test_frontend_routes_exist_in_backend[agent-framework]" -q`
Expected: PASS (the page references `styles.css`/`audio.js`/`app.js`, all present; it fetches no `/api/*` path — it uses a WebSocket — so the route check is satisfied trivially).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/init/templates/agent-framework/static/index.html aai_cli/init/templates/agent-framework/static/app.js
git commit -m "feat(agent-framework): frontend (cascade UI + /ws client)"
```

---

## Task 8: Scaffold parity files (deploy + docs + deps)

**Files (all create under `aai_cli/init/templates/agent-framework/`):**
- `requirements.txt`, `env.example`, `gitignore`, `runtime.txt`, `vercel.json`, `Procfile`, `Dockerfile`, `dockerignore`, `README.md`, `AGENTS.md`

- [ ] **Step 1: `requirements.txt`**

```text
fastapi>=0.136.3
uvicorn>=0.30.0
websockets>=14.1
openai>=1.54.0
python-dotenv>=1.2.2
# Pin starlette directly: FastAPI's own floor still admits versions with known CVEs,
# so raise the transitive floor above them.
starlette>=1.2.1
```

(`websockets` uses `additional_headers`, supported from 14.x; if the `install` test reports it unsupported, bump the floor. `openai>=1.54.0` provides `AsyncOpenAI` + streamed `chat.completions`.)

- [ ] **Step 2: `env.example`**

```text
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
# This cascade uses streaming TTS, which is sandbox-only — use a sandbox key and the
# sandbox hosts (assembly --sandbox init agent-framework fills these in for you):
# ASSEMBLYAI_STREAMING_HOST=streaming.sandbox000.assemblyai-labs.com
# ASSEMBLYAI_TTS_HOST=streaming-tts.sandbox000.assemblyai-labs.com
# ASSEMBLYAI_LLM_GATEWAY_URL=https://llm-gateway.sandbox000.assemblyai-labs.com/v1
```

- [ ] **Step 3: `gitignore`, `runtime.txt`, `vercel.json`, `dockerignore` (copy voice-agent's shapes)**

Run:

```bash
SRC=aai_cli/init/templates/voice-agent
DST=aai_cli/init/templates/agent-framework
cp "$SRC/gitignore" "$DST/gitignore"
cp "$SRC/runtime.txt" "$DST/runtime.txt"
cp "$SRC/vercel.json" "$DST/vercel.json"
cp "$SRC/dockerignore" "$DST/dockerignore"
```

- [ ] **Step 4: `Procfile`**

```text
web: python -m uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-3000}
```

- [ ] **Step 5: `Dockerfile` (copy voice-agent's — it already satisfies the contract: EXPOSE 8080, `${PORT:-8080}`, non-root USER)**

Run: `cp aai_cli/init/templates/voice-agent/Dockerfile aai_cli/init/templates/agent-framework/Dockerfile`

- [ ] **Step 6: `README.md`**

```markdown
# Talk to a cascaded voice agent — AssemblyAI agent-framework starter

Click connect and talk. Unlike the `voice-agent` template (which uses AssemblyAI's
all-in-one Voice Agent API), this app is a **cascade your own backend orchestrates**:
Streaming STT transcribes you, the LLM Gateway generates a reply, and streaming TTS
speaks it back — with turn detection and barge-in handled server-side. The browser
holds one WebSocket to your backend, so your API key never reaches the client.

## Sandbox-only

Streaming TTS has no production host, so the whole cascade runs against the AssemblyAI
sandbox with a sandbox key. Scaffold it that way:

```sh
assembly --sandbox init agent-framework
```

That pins the sandbox hosts in `.env`. Running against production exits with a hint.

## Run locally

```sh
assembly dev   # opens http://localhost:3000 (allow microphone access; headphones recommended)
```

`ASSEMBLYAI_API_KEY` is read from `.env` (created for you by `assembly init`).

## Deploy

This app keeps a **long-running WebSocket**, so it needs a persistent process — not
Vercel's serverless functions. Use the shipped `Procfile`/`Dockerfile` on Render,
Railway, Fly.io, or Google Cloud Run (`gcloud run deploy --source .`):

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

Set `ASSEMBLYAI_API_KEY` and the three sandbox host vars (`ASSEMBLYAI_STREAMING_HOST`,
`ASSEMBLYAI_TTS_HOST`, `ASSEMBLYAI_LLM_GATEWAY_URL`) in the platform's environment.

## Ideas to extend

- Change the `MODEL`, `VOICE`, `SYSTEM_PROMPT`, or `GREETING` in `api/settings.py`.
- Stream each LLM sentence into TTS as it completes (lower latency) instead of
  synthesizing the whole reply at once — see `_generate_reply` in `api/cascade.py`.
- Add tools (function calling) on the LLM leg so the agent can look things up.
```

- [ ] **Step 7: `AGENTS.md` (must contain `ASSEMBLYAI_API_KEY`, `buildless`, `static/app.js` for the contract)**

```markdown
# Agent Notes

This is a buildless FastAPI + browser starter for a **cascaded** voice agent
(Streaming STT -> LLM Gateway -> streaming TTS), orchestrated server-side. Run it with:

```sh
assembly dev
```

## Map

- `api/settings.py`: API key, hosts, model, voice, system prompt, greeting, sample rates.
- `api/cascade.py`: the orchestrator — STT/TTS socket helpers, the LLM stream, turn
  detection, barge-in, and the `/ws` browser adapter. Built with injected `Deps` so it
  is tested against fakes.
- `api/index.py`: FastAPI app — serves the page/assets and the `/ws` WebSocket.
- `static/app.js`: WebSocket lifecycle, mic capture, UI state, and event handling
  (`_CONFIG` block at the top is the primary edit point).
- `static/audio.js`: microphone pipeline, PCM conversion, playback queue, barge-in.
- `static/styles.css`: visual styling only; the top `:root` block is the theme edit point.
- `static/index.html`: page structure and static asset links.

## Change Points

- Model, voice, prompt, greeting, sample rates: edit `api/settings.py`.
- Cascade behavior (turn detection, barge-in, LLM->TTS piping): edit `api/cascade.py`.
- Transcript log rendering: edit `addTurn` in `static/app.js`.
- Playback, barge-in, or PCM conversion: edit `static/audio.js`.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `static/`.
- Streaming TTS is sandbox-only; keep this app pointed at the sandbox hosts.
- `reply.audio` carries base64 PCM on the `data` field.
- The browser ↔ backend event protocol matches the `voice-agent` template — keep it
  stable so `static/audio.js` and the UI stay reusable.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
```

- [ ] **Step 8: Run the full parametrized contract suite for this template**

Run: `uv run pytest tests/test_init_template_contract.py tests/test_init_template_serve.py -q -k agent-framework`
Expected: PASS for every parametrized case (`agent-framework`): required files, vercel framework pin, Dockerfile shape, dockerignore `.env`, no `public/`, Procfile, runtime pin, static refs, AGENTS edit points, no committed dotenv, requirements cover imports + pinned, root + static assets served.

- [ ] **Step 9: Commit**

```bash
git add aai_cli/init/templates/agent-framework/requirements.txt aai_cli/init/templates/agent-framework/env.example aai_cli/init/templates/agent-framework/gitignore aai_cli/init/templates/agent-framework/runtime.txt aai_cli/init/templates/agent-framework/vercel.json aai_cli/init/templates/agent-framework/Procfile aai_cli/init/templates/agent-framework/Dockerfile aai_cli/init/templates/agent-framework/dockerignore aai_cli/init/templates/agent-framework/README.md aai_cli/init/templates/agent-framework/AGENTS.md
git commit -m "feat(agent-framework): deploy, docs, and dependency scaffold"
```

---

## Task 9: Regenerate snapshots + full gate

**Files:**
- Modify: `tests/__snapshots__/test_snapshots_help_build.ambr` (regenerated)

- [ ] **Step 1: Regenerate the `--help` snapshots (the init arg help now lists the new template)**

Run: `uv run pytest tests/test_snapshots_help_build.py --snapshot-update -q`
Then review: `git diff tests/__snapshots__/test_snapshots_help_build.ambr`
Expected: the only change is `agent-framework` appended to the `init` template enumeration. If other help snapshot files changed, regenerate them too (`uv run pytest -k snapshots_help --snapshot-update`).

- [ ] **Step 2: Run the targeted suites green**

Run: `uv run pytest tests/test_init_template_agent_framework.py tests/test_init_templates.py tests/test_init_command.py tests/test_init_template_contract.py tests/test_init_template_serve.py -q`
Expected: all PASS.

- [ ] **Step 3: Run the install smoke test for this template (network + uv required)**

Run: `uv run pytest -m install -q -k agent-framework`
Expected: PASS (requirements install into a clean venv and `api.index` imports). If `websockets`/`openai` floors are wrong, bump them in `requirements.txt` and re-run.

- [ ] **Step 4: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` Watch specifically for:
- `prettier` (template JS/CSS) — clean.
- `ruff`/`ruff format` over `api/*.py` — clean.
- `diff-cover` 100% patch coverage — every new `cascade.py`/`index.py`/`settings.py` line is covered by Task 3–6 tests. If a line is reported uncovered, add a direct assertion (do not add `pragma: no cover` for reachable orchestration lines).
- mutation gate — a surviving mutant means a changed line lacks a *failing-on-break* assertion; strengthen the relevant test.
- the init template contract gate + unused snapshot/fixture gate.

- [ ] **Step 5: Commit the regenerated snapshot (only if not already committed) and finalize**

```bash
git add tests/__snapshots__/test_snapshots_help_build.ambr
git commit -m "test(init): refresh --help snapshot for agent-framework template"
```

---

## Self-review notes (resolved)

- **Spec coverage:** every spec section maps to a task — architecture/orchestrator (T4–T6), components (T2–T8), CLI wiring (T1), deploy/sandbox caveats (T8 README/AGENTS/settings guard), error handling (T5 `session.error` paths), testing (T3–T9).
- **Import-time safety:** `settings.py` never raises (T3 test `test_settings_imports_without_key_or_tts_host`); the availability guard is in `run_session` (T5).
- **Coverage/mutation burden:** orchestrator is decomposed into directly-testable units (`unavailable_reason`, `stt_url`, `tts_url`, `is_final_user_turn`, `build_messages`, `_pump_mic`, `_synthesize`, `_speak`, `_generate_reply`, `maybe_barge_in`, `_pump_stt`, `run_session`, `FastAPIBrowser`), each with an asserting test.
- **Naming consistency:** `Deps`, `Session`, `run_session`, `_synthesize`, `_generate_reply`, `_speak`, `_pump_mic`, `_pump_stt`, `maybe_barge_in`, `FastAPIBrowser` used identically across tasks and tests.
- **Gated assertions:** the exact help-string test (T1) and the `--help` snapshot (T9) are both updated for the new registry entry.
```
