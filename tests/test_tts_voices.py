"""Tests for the streaming-TTS voice catalog (aai_cli/tts/voices.py): the
voice -> language mapping and the language-driven default-voice selection
`assembly speak` and `assembly dub` share."""

from __future__ import annotations

import pytest

from aai_cli.tts import voices


def test_voice_languages_catalog():
    # An independent copy of the expected catalog: a silently edited entry in
    # the shipped map must fail here, not just round-trip through itself.
    assert voices.VOICE_LANGUAGES == {
        "alba": "en",
        "anna": "en",
        "azelma": "en",
        "bill_boerst": "en",
        "caro_davy": "en",
        "charles": "en",
        "cosette": "en",
        "eponine": "en",
        "estelle": "fr",
        "eve": "en",
        "fantine": "en",
        "george": "en",
        "giovanni": "it",
        "jane": "en",
        "javert": "en",
        "jean": "en",
        "juergen": "de",
        "lola": "es",
        "marius": "en",
        "mary": "en",
        "michael": "en",
        "paul": "en",
        "peter_yearsley": "en",
        "rafael": "pt",
        "stuart_bell": "en",
        "vera": "en",
    }


def test_english_rotation_is_the_confirmed_working_voices():
    assert voices.ENGLISH_ROTATION == ("jane", "michael", "mary", "paul", "eve", "george")
    # Every rotation voice must actually speak English.
    assert all(voices.VOICE_LANGUAGES[voice] == "en" for voice in voices.ENGLISH_ROTATION)


def test_every_voice_language_has_a_name():
    # rotation_for relies on this: any code a catalog voice speaks must
    # normalize through language_code, so a rotation can never come back empty.
    for code in set(voices.VOICE_LANGUAGES.values()):
        assert voices.language_code(code) == code
        assert voices.rotation_for(code)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("de", "de"),  # an ISO code passes through
        (" DE ", "de"),  # trimmed and case-insensitive
        ("German", "de"),  # a language name maps to its code
        ("english", "en"),
        ("Portuguese", "pt"),
        (None, None),  # no language requested
        ("Klingon", None),  # unknown language -> no catalog voices
        ("ja", None),  # a translatable language without a voice
    ],
)
def test_language_code(value, expected):
    assert voices.language_code(value) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("English", voices.ENGLISH_ROTATION),
        ("en", voices.ENGLISH_ROTATION),
        (None, voices.ENGLISH_ROTATION),  # server-default language -> English voices
        ("Japanese", voices.ENGLISH_ROTATION),  # no native voice -> English fallback
        ("Italian", ("giovanni",)),
        ("es", ("lola",)),
        ("German", ("juergen",)),
        ("pt", ("rafael",)),
        ("French", ("estelle",)),
    ],
)
def test_rotation_for(language, expected):
    assert voices.rotation_for(language) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [("English", "jane"), ("Italian", "giovanni"), ("fr", "estelle"), ("Klingon", "jane")],
)
def test_default_voice(language, expected):
    assert voices.default_voice(language) == expected
