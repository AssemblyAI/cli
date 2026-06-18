"""Tests for the cascade's per-run configuration defaults."""

from __future__ import annotations

import dataclasses

import pytest

from aai_cli.agent_cascade.config import (
    DEFAULT_GREETING,
    DEFAULT_MAX_HISTORY,
    DEFAULT_MODEL,
    CascadeConfig,
)
from aai_cli.agent_cascade.voices import DEFAULT_VOICE
from aai_cli.core import llm


def test_default_config_values():
    config = CascadeConfig()
    assert config.voice == DEFAULT_VOICE
    assert config.model == DEFAULT_MODEL == "gpt-5.1"  # `assembly live` defaults to gpt-5.1
    assert config.greeting == DEFAULT_GREETING
    # The sliding-window default keeps the last 40 messages of context.
    assert config.max_history == 40
    assert DEFAULT_MAX_HISTORY == 40
    # Formatting is on by default, so the reply trigger waits for the formatted turn.
    assert config.format_turns is True
    assert config.language is None
    assert config.max_tokens == llm.DEFAULT_MAX_TOKENS
    # Escape-hatch overrides start empty.
    assert dict(config.llm_extra) == {}
    assert dict(config.tts_extra) == {}


def test_config_is_frozen():
    # Frozen so a parsed run config can't be mutated mid-conversation.
    config = CascadeConfig()
    attr = "voice"  # not a literal, so ruff's B010 leaves the setattr in place
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, attr, "other")
