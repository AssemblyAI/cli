# aai_cli/init/steps.py
from __future__ import annotations

from typing import TypedDict

from rich.markup import escape

from aai_cli import theme


class Step(TypedDict):
    name: str
    status: str
    detail: str


def render_steps(items: list[Step]) -> str:
    lines = []
    for s in items:
        style = theme.status_style(s["status"])
        lines.append(
            f"  {escape(s['name'])}: "
            f"[{style}]{escape(s['status'])}[/{style}] — {escape(s['detail'])}"
        )
    return "[aai.heading]aai init:[/aai.heading]\n" + "\n".join(lines)
