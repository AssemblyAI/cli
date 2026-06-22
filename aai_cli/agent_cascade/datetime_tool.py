"""A keyless local date/time tool for the `assembly live` voice agent.

Reports the current local date and time so the live agent can answer "what time is
it?", "what's today's date?", or "what day is it?". It needs no network and no API
key — just the system clock — making it, like the weather tool, always present.

The only non-determinism is the :data:`Clock` seam (a ``() -> datetime`` callable),
injected in tests so the flow is deterministic with no real clock. Everything else
(the spoken formatting) is pure and tested directly. There is no failure mode to
handle: reading the local clock cannot fail, so the tool returns unconditionally.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# The registered tool name. ``brain.py`` keys its UI label and capability phrase off
# this, so a test pins it.
DATETIME_TOOL_NAME = "get_current_datetime"

# A clock returns the current instant. Injected in tests (the only non-determinism).
Clock = Callable[[], datetime]


def _now() -> datetime:
    """Return the current local time as a timezone-aware datetime (the default clock)."""
    return datetime.now().astimezone()


def _format(now: datetime) -> str:
    """Render ``now`` as one short, speakable date+time string.

    Uses only cross-platform ``strftime`` codes (no ``%-d``/``%-I``, which break on
    Windows). Zero-padded day/hour is fine — the model reads the string aloud.
    """
    return now.strftime("It's %A, %B %d, %Y at %I:%M %p %Z.")


def build_datetime_tool(now: Clock = _now) -> BaseTool:
    """Wrap the local clock as the ``get_current_datetime`` tool (``now`` injectable)."""
    from langchain_core.tools import tool

    @tool(DATETIME_TOOL_NAME)
    def get_current_datetime() -> str:
        """Get the current local date and time. Use when asked the date, the day of the
        week, or the time."""
        return _format(now())

    return get_current_datetime
