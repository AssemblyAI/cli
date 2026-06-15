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
