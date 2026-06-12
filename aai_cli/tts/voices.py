"""The streaming-TTS voice catalog: every voice speaks exactly one language.

When no ``--voice`` is chosen, `assembly speak` and `assembly dub` pick the
voice from the requested language: a non-English language switches to that
language's native voice(s) — most ship exactly one, so the language alone
selects the voice — while English keeps the curated multi-speaker rotation.
"""

from __future__ import annotations

# Voice id -> ISO 639-1 code of the (single) language the voice speaks.
VOICE_LANGUAGES: dict[str, str] = {
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

# The language names the TTS `language` param uses, keyed by ISO code. Only
# languages with at least one catalog voice belong here (rotation_for relies
# on that invariant to never resolve to an empty rotation). Deliberately
# narrower than dub_exec.LANGUAGE_NAMES, which also lists voiceless languages
# the translator supports.
_LANGUAGE_NAMES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
}

_NAME_TO_CODE = {name.casefold(): code for code, name in _LANGUAGE_NAMES.items()}

# English has many voices; this curated subset keeps multi-speaker output
# varied with the confirmed-working voices. Non-English languages rotate
# through their own (usually single) native voices instead.
ENGLISH_ROTATION = ("jane", "michael", "mary", "paul", "eve", "george")


def language_code(language: str | None) -> str | None:
    """Normalize a language value (ISO code or name, any case) to its code,
    or None when it has no catalog voices — the TTS service and the dub
    translator accept more languages than the catalog covers."""
    if language is None:
        return None
    cleaned = language.strip().casefold()
    if cleaned in _LANGUAGE_NAMES:
        return cleaned
    return _NAME_TO_CODE.get(cleaned)


def rotation_for(language: str | None) -> tuple[str, ...]:
    """The default voice rotation for a language.

    English — and any language without catalog voices — keeps the curated
    English rotation; a language with native voices rotates through those, so
    a single-voice language always switches to its one voice.
    """
    code = language_code(language)
    if code is None or code == "en":
        return ENGLISH_ROTATION
    return tuple(voice for voice, spoken in VOICE_LANGUAGES.items() if spoken == code)


def default_voice(language: str | None) -> str:
    """The voice used when none is chosen: the language's first rotation voice."""
    return rotation_for(language)[0]
