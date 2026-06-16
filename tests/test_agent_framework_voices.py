"""Tests for the cascade's voice catalog presentation."""

from __future__ import annotations

from aai_cli.agent_framework import voices
from aai_cli.tts import voices as tts_voices


def test_voice_names_are_the_tts_catalog_sorted():
    assert sorted(tts_voices.VOICE_LANGUAGES) == voices.VOICE_NAMES


def test_default_voice_is_in_catalog():
    assert voices.DEFAULT_VOICE in voices.VOICE_NAMES


def test_complete_voice_filters_by_prefix():
    completions = voices.complete_voice("ja")
    assert "jane" in completions
    assert all(name.startswith("ja") for name in completions)


def test_format_voice_list_groups_by_language():
    listing = voices.format_voice_list()
    blocks = {block.split(":", 1)[0]: block for block in listing.split("\n\n")}
    # Each voice is filed strictly under the language it actually speaks: the English
    # block lists jane but not the Italian-only giovanni, and vice versa.
    assert "jane" in blocks["English"]
    assert "giovanni" not in blocks["English"]
    assert "giovanni" in blocks["Italian"]
    assert "jane" not in blocks["Italian"]
