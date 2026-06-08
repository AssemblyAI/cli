from __future__ import annotations

from aai_cli import output

GOAL = 100

# Counts that earn a one-off cheer; keep keys in sync with the wizard's nudge.
_MILESTONES: dict[int, str] = {
    1: "You're activated 🎉 — your first request is in.",
    10: "10 requests in. You're getting the hang of it.",
    50: "Halfway to 100 — nice momentum.",
    GOAL: f"{GOAL} requests — you're off the ground. 🚀",
}


def milestone_message(count: int) -> str | None:
    """Encouragement to show when a request count lands exactly on a milestone."""
    return _MILESTONES.get(count)


def render_progress(count: int) -> str:
    """A Rich-markup block: 'N of 100 API requests', any milestone, the usage pointer."""
    lines = [output.success(f"{count} of {GOAL} API requests")]
    cheer = milestone_message(count)
    if cheer:
        lines.append("  " + output.heading(cheer))
    lines.append("  " + output.hint("For your full account usage, run `aai usage`."))
    return "\n".join(lines)
