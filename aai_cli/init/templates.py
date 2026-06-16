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


# One-line description shown beside each title in the interactive picker. Keys must
# match TEMPLATES exactly (a test enforces both directions).
DESCRIPTIONS: dict[str, str] = {
    "audio-transcription": "Transcribe audio & video files, URLs, and YouTube — speaker labels and audio intelligence",
    "live-captions": "Live real-time captions from your microphone over the Streaming API",
    "voice-agent": "Full-duplex voice agent (speech in, LLM reply, speech out) via the Voice Agent API",
    "agent-framework": "Cascaded voice agent you orchestrate: Streaming STT, the LLM Gateway, and sandbox TTS",
}


def dir_for(name: str) -> str:
    """The on-disk template directory for an id: kebab id -> underscore package dir."""
    return name.replace("-", "_")


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)


def description_for(name: str) -> str:
    """The one-line picker description for a template id, or '' when unknown."""
    return DESCRIPTIONS.get(name, "")
