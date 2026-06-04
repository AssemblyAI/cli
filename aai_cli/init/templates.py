from __future__ import annotations

# id -> human-facing title shown in the picker. Ids mirror the CLI's commands.
TEMPLATES: dict[str, str] = {
    "transcribe": "Transcribe & explore a file",
    "stream": "Live captions (mic → browser)",
    "agent": "Talk to a voice agent",
    "llm": "Chat with your audio (LLM)",
}

# Display order for the picker and `--help`, matching the CLI's command order.
TEMPLATE_ORDER: tuple[str, ...] = ("transcribe", "stream", "agent", "llm")


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)
