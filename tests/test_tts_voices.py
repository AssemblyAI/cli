"""Tests for the streaming-TTS voice catalog (aai_cli/tts/voices.py): the built-in
voices (each speaks every language), the language names the `language` connect
parameter takes, and the voice_id mapping `speak`, `dub`, and `live` share."""

from __future__ import annotations

import pytest

from aai_cli.tts import voices


def test_builtin_voices_are_the_documented_set():
    # An independent copy of the documented built-in voices (/docs/tts/voices): a
    # silently edited entry in the shipped tuple must fail here.
    assert voices.VOICES == (
        "alba",
        "anna",
        "charles",
        "eve",
        "george",
        "jane",
        "jean",
        "mary",
        "michael",
        "paul",
        "vera",
    )


def test_languages_are_the_documented_set():
    assert voices.LANGUAGES == {
        "de": "german",
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "it": "italian",
        "pt": "portuguese",
    }


def test_defaults_are_builtin_voices():
    assert voices.DEFAULT_VOICE == "jane"
    assert voices.DEFAULT_ROTATION == ("jane", "michael", "mary", "paul", "eve", "george")
    assert set(voices.DEFAULT_ROTATION) <= set(voices.VOICES)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "de"),  # an ISO code passes through
        (" DE ", "de"),  # trimmed and case-insensitive
        ("German", "de"),  # a language name maps to its code
        ("english", "en"),
        ("Portuguese", "pt"),
        (None, None),  # no language requested
        ("Klingon", None),  # a language the service doesn't speak
        ("ja", None),
    ],
)
def test_language_code(value, expected):
    assert voices.language_code(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "german"),  # the service takes the English name, not the code
        ("German", "german"),  # names are lowercased
        (" FR ", "french"),
        ("english", "english"),
        ("Japanese", "japanese"),  # unsupported: passes through for the server to refuse
    ],
)
def test_language_param(value, expected):
    assert voices.language_param(value) == expected


@pytest.mark.parametrize(
    ("voice", "expected"),
    [
        ("jane", "aai-jane"),  # a built-in name gets its library slug
        ("vera", "aai-vera"),
        ("aai-jane", "aai-jane"),  # a slug copied from --list-voices passes through
        ("alex-6f2f2e", "alex-6f2f2e"),  # an account (cloned) voice slug is sent as-is
        ("01a160a3-8ebb-784f-92b0-6927026f2f2e", "01a160a3-8ebb-784f-92b0-6927026f2f2e"),
    ],
)
def test_voice_id(voice, expected):
    assert voices.voice_id(voice) == expected


@pytest.mark.parametrize(
    ("voice", "expected"),
    [("aai-jane", "jane"), ("jane", "jane"), ("aai-aai-x", "aai-x")],  # one prefix, not all
)
def test_voice_name(voice, expected):
    assert voices.voice_name(voice) == expected
