"""The voices `assembly agent-cascade` speaks with.

The cascade's audio comes from streaming TTS, so its voices are the TTS catalog
(`aai_cli.tts.voices`) — not the Voice Agent voices `assembly agent` uses. This
module is the thin presentation layer over that catalog: the membership list
that catches a typo'd ``--voice`` and the completion callback. (``--list-voices``
itself is the service's live listing — see ``app.tts_voices``.)
"""

from __future__ import annotations

from aai_cli.core import choices
from aai_cli.tts import voices as tts_voices

DEFAULT_VOICE = "jane"

# The selectable voice names, sorted for a stable completion order.
VOICE_NAMES: list[str] = sorted(tts_voices.VOICE_LANGUAGES)


def complete_voice(incomplete: str) -> list[str]:
    """Shell-completion callback for ``--voice``: catalog names matching the prefix."""
    return choices.complete_prefix(VOICE_NAMES, incomplete)
