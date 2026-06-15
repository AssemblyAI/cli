"""Hermetic tests for the agent-framework (cascaded voice agent) template.

The template ships a standalone FastAPI app under api/; load it by path with its
own `api` package, evicting any other template's cached `api` modules so imports
stay collision-free under pytest-xdist / pytest-randomly.
"""

from __future__ import annotations

import importlib
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
