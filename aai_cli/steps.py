from __future__ import annotations

from typing import TypedDict

from rich.markup import escape

from aai_cli import theme


class Step(TypedDict):
    """One line of multi-step command output: a named step, its status, and a detail."""

    name: str
    status: str
    detail: str


def render_steps(items: list[Step], *, heading: str) -> str:
    """Render steps as a themed heading followed by one status-styled line each.

    Shared by the multi-step commands (`aai init`, `aai setup`); each passes its
    own heading.
    """
    lines: list[str] = []
    for s in items:
        style = theme.status_style(s["status"])
        lines.append(
            f"  {escape(s['name'])}: "
            f"[{style}]{escape(s['status'])}[/{style}] — {escape(s['detail'])}"
        )
    return f"[aai.heading]{escape(heading)}[/aai.heading]\n" + "\n".join(lines)
