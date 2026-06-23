"""Read project-instruction files (``AGENTS.md``/``CLAUDE.md``) into the live agent's context.

`assembly live` runs in the user's working directory, so — like a coding agent — it reads the
project's instruction files into its system prompt when present, giving spoken answers grounded
in the project it's launched from. ``AGENTS.md`` is the cross-agent standard and ``CLAUDE.md`` is
frequently a symlink to it, so identical content is included once, and the total is capped so an
oversized instructions file can't crowd the conversation out of the model's window.
"""

from __future__ import annotations

from pathlib import Path

# The instruction files an agentic CLI reads into context, highest precedence first.
CONTEXT_FILENAMES = ("AGENTS.md", "CLAUDE.md")

# Cap the injected context: the spoken agent only needs the project's gist, and an unusually
# large instructions file would otherwise crowd the live conversation out of the model's window.
# A +-1 shift in the budget is behaviorally equivalent, so no test can kill a mutant on it.
MAX_CONTEXT_CHARS = 16000  # pragma: no mutate

# Appended when the content is truncated, so the model knows it's seeing only the head of the file.
_TRUNCATION_MARKER = "\n\n[project context truncated]"


def _read_instructions(path: Path) -> str | None:
    """The stripped contents of one instruction file, or ``None`` if absent/unreadable/empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _truncate(combined: str) -> str:
    """Cap the combined context at :data:`MAX_CONTEXT_CHARS`, marking it when truncated."""
    if len(combined) > MAX_CONTEXT_CHARS:
        return combined[:MAX_CONTEXT_CHARS] + _TRUNCATION_MARKER
    return combined


def load_project_context(directory: Path | None = None) -> str | None:
    """Read the project-instruction files in *directory* into one de-duplicated string.

    Looks for each name in :data:`CONTEXT_FILENAMES` under *directory* (the current working
    directory by default), returning their stripped contents joined under a per-file heading —
    or ``None`` when none are present, readable, or non-empty. Identical files (``CLAUDE.md`` is
    commonly a symlink to ``AGENTS.md``) are included once, and the combined text is truncated to
    :data:`MAX_CONTEXT_CHARS` so a huge file can't crowd out the live conversation.
    """
    base = Path.cwd() if directory is None else directory
    sections: list[str] = []
    seen: set[str] = set()
    for name in CONTEXT_FILENAMES:
        text = _read_instructions(base / name)
        if text is None or text in seen:
            continue
        seen.add(text)
        sections.append(f"# {name}\n\n{text}")
    if not sections:
        return None
    return _truncate("\n\n".join(sections))
