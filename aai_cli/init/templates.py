from __future__ import annotations

# id -> human-facing title shown in the picker. Ids mirror the CLI's commands.
#
# Every id here MUST have a directory under templates/<id>/ (a test enforces both
# directions) — the picker must never advertise a template that would crash on scaffold.
TEMPLATES: dict[str, str] = {
    "transcribe": "Transcribe a pre-recorded file (+ chat with it via LLM Gateway)",
    "stream": "Live captions (mic → browser)",
    "agent": "Talk to a voice agent",
}

# Display order for the picker and `--help`, matching the CLI's command order.
TEMPLATE_ORDER: tuple[str, ...] = ("transcribe", "stream", "agent")


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)
