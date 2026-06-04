from __future__ import annotations

# id -> human-facing title shown in the picker. Ids mirror the CLI's commands.
#
# Only list templates whose directory actually ships under templates/<id>/ — the
# picker must never advertise a template that would crash on scaffold. The design
# covers four (transcribe, stream, agent, llm); the latter three are added here as
# their template directories land in follow-on PRs. A test enforces registry ==
# shipped directories.
TEMPLATES: dict[str, str] = {
    "transcribe": "Transcribe & explore a file",
}

# Display order for the picker and `--help`.
TEMPLATE_ORDER: tuple[str, ...] = ("transcribe",)


def is_template(name: str) -> bool:
    return name in TEMPLATES


def title_for(name: str) -> str:
    """The human title for a template id, or the raw id if unknown."""
    return TEMPLATES.get(name, name)
