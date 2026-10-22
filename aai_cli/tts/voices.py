"""The streaming-TTS voice catalog: the service's built-in voices and languages.

The built-in voices (documented at /docs/tts/voices) all speak every supported
language — the session's ``language`` connect parameter picks the language,
not the voice — so the default voice and the multi-speaker rotation are the
same whatever the language.

On the wire a built-in voice is addressed by its library slug (``aai-<name>``)
as ``voice_id``; the bare ``voice=<name>`` parameter is retired. Every other
value — a voice's UUID or one of the account's own (e.g. cloned) voice slugs —
is sent as given. ``--list-voices`` adds the account's own voices from the
service's ``GET /v1/voices`` — see ``voice_list``.
"""

from __future__ import annotations

# The slug prefix of the service's own library voices.
VOICE_ID_PREFIX = "aai-"

# The built-in voices; each speaks every language in LANGUAGES.
VOICES: tuple[str, ...] = (
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

# The languages the service speaks, keyed by ISO 639-1 code. The `language` connect
# parameter takes the lowercase English name, not the code.
LANGUAGES: dict[str, str] = {
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "it": "italian",
    "pt": "portuguese",
}

_NAME_TO_CODE = {name: code for code, name in LANGUAGES.items()}

# The voice used when none is chosen (the service's own default, too).
DEFAULT_VOICE = "jane"

# A curated subset that keeps multi-speaker output varied; any language.
DEFAULT_ROTATION = ("jane", "michael", "mary", "paul", "eve", "george")


def voice_id(voice: str) -> str:
    """The ``voice_id`` for a voice: a built-in name gets its library slug
    (``jane`` -> ``aai-jane``); anything else (``aai-jane``, a UUID, or one of
    the account's own voice slugs) is already a ``voice_id`` and passes through."""
    return f"{VOICE_ID_PREFIX}{voice}" if voice in VOICES else voice


def voice_name(voice: str) -> str:
    """The catalog name behind a ``--voice`` value: ``aai-jane`` -> ``jane``, so a
    copied ``voice_id`` validates against the catalog like the bare name does."""
    return voice.removeprefix(VOICE_ID_PREFIX)


def language_code(language: str | None) -> str | None:
    """Normalize a language value (ISO code or name, any case) to its code, or
    None when the service doesn't speak it."""
    if language is None:
        return None
    cleaned = language.strip().casefold()
    if cleaned in LANGUAGES:
        return cleaned
    return _NAME_TO_CODE.get(cleaned)


def language_param(language: str) -> str:
    """The ``language`` connect parameter for a language value: the lowercase
    English name (``de`` / ``German`` -> ``german``). A language the service
    doesn't speak passes through lowercased, so the server's own 3006 error —
    which lists the supported set — reaches the user."""
    code = language_code(language)
    return LANGUAGES[code] if code is not None else language.strip().casefold()
