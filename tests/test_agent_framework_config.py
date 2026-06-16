"""Tests for the cascade's per-run configuration defaults."""

from __future__ import annotations

import dataclasses

import pytest

from aai_cli.agent_framework.config import DEFAULT_GREETING, DEFAULT_MAX_HISTORY, CascadeConfig
from aai_cli.agent_framework.voices import DEFAULT_VOICE
from aai_cli.core import llm


def test_default_config_values():
    config = CascadeConfig()
    assert config.voice == DEFAULT_VOICE
    assert config.model == llm.DEFAULT_MODEL
    assert config.greeting == DEFAULT_GREETING
    # The sliding-window default keeps the last 40 messages of context.
    assert config.max_history == 40
    assert DEFAULT_MAX_HISTORY == 40


def test_config_is_frozen():
    # Frozen so a parsed run config can't be mutated mid-conversation.
    config = CascadeConfig()
    attr = "voice"  # not a literal, so ruff's B010 leaves the setattr in place
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, attr, "other")
