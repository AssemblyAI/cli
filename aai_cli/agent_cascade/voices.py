"""The voices `assembly agent-cascade` speaks with.

The cascade's audio comes from streaming TTS, so its voices are the TTS catalog
(`aai_cli.tts.voices`) — not the Voice Agent voices `assembly agent` uses. This
module is the thin presentation layer over that catalog: the membership list
that catches a typo'd ``--voice``, the completion callback, and the grouped
``--list-voices`` rendering.
"""

from __future__ import annotations

from aai_cli.core import choices
from aai_cli.tts import voices as tts_voices

DEFAULT_VOICE = "jane"

# The selectable voice ids, sorted for a stable --list-voices / completion order.
VOICE_NAMES: list[str] = sorted(tts_voices.VOICE_LANGUAGES)

# ISO 639-1 code -> the heading --list-voices groups that language's voices under.
_LANGUAGE_LABELS: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
}


def complete_voice(incomplete: str) -> list[str]:
    """Shell-completion callback for ``--voice``: catalog ids matching the prefix."""
    return choices.complete_prefix(VOICE_NAMES, incomplete)


def format_voice_list() -> str:
    """Human-readable voice ids for ``--list-voices``, grouped by language."""
    blocks: list[str] = []
    for code, label in _LANGUAGE_LABELS.items():
        names = [name for name in VOICE_NAMES if tts_voices.VOICE_LANGUAGES[name] == code]
        if names:
            listing = "\n".join(f"  {name}" for name in names)
            blocks.append(f"{label}:\n{listing}")
    return "\n\n".join(blocks)
