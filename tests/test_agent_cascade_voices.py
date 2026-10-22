"""Tests for the cascade's voice catalog presentation."""

from __future__ import annotations

import dataclasses

import pytest

from aai_cli.agent_cascade import voices
from aai_cli.app.context import AppState
from aai_cli.commands.agent_cascade._exec import run_agent_cascade
from aai_cli.core.errors import CLIError, UsageError
from aai_cli.tts import voices as tts_voices
from tests.test_agent_cascade_command import _DEFAULTS


def test_voice_names_are_the_tts_catalog_sorted():
    assert sorted(tts_voices.VOICE_LANGUAGES) == voices.VOICE_NAMES


def test_default_voice_is_in_catalog():
    assert voices.DEFAULT_VOICE in voices.VOICE_NAMES


def test_complete_voice_filters_by_prefix():
    completions = voices.complete_voice("ja")
    assert "jane" in completions
    assert all(name.startswith("ja") for name in completions)


def test_voice_id_form_validates_as_its_catalog_name():
    # A voice_id copied from --list-voices (aai-jane) is the catalog voice jane, so it
    # passes validation and the run proceeds to the next guard (the sandbox check).
    with pytest.raises(CLIError, match="sandbox"):
        run_agent_cascade(
            dataclasses.replace(_DEFAULTS, voice="aai-jane"), AppState(), json_mode=False
        )
    with pytest.raises(UsageError, match="Unknown voice"):
        run_agent_cascade(
            dataclasses.replace(_DEFAULTS, voice="aai-nope"), AppState(), json_mode=False
        )
