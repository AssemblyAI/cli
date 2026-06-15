from __future__ import annotations

# id -> human-facing title shown in the picker. Ids are Vercel-style
# project/example slugs rather than CLI command names.
#
# Every id here MUST have a directory under templates/<id>/ (a test enforces both
# directions) — the picker must never advertise a template that would crash on scaffold.
TEMPLATES: dict[str, str] = {
    "audio-transcription": "Audio Transcription",
    "live-captions": "Live Captions",
    "voice-agent": "Voice Agent",
    "agent-framework": "Agent Framework",
}

# Display order for the picker and `--help`.
TEMPLATE_ORDER: tuple[str, ...] = (
    "audio-transcription",
    "live-captions",
    "voice-agent",
    "agent-framework",
)


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)
