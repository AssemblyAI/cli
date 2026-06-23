"""Tests for the keyless local date/time tool behind `assembly live`.

The tool's only non-determinism is the injected ``Clock`` callable, so the whole
flow is deterministic with no real clock (and pytest-socket stays armed — no I/O).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aai_cli.agent_cascade import datetime_tool

# A fixed, timezone-aware instant: Monday, 2026-06-22 14:30 at a fixed -07:00 offset.
# A fixed offset (not a named zone) keeps %Z deterministic cross-platform without tzdata.
_FIXED = datetime(2026, 6, 22, 14, 30, tzinfo=timezone(timedelta(hours=-7)))
_EXPECTED = "It's Monday, June 22, 2026 at 02:30 PM UTC-07:00."


# --- _format -----------------------------------------------------------------


def test_format_renders_exact_speakable_string():
    assert datetime_tool._format(_FIXED) == _EXPECTED


# --- _now (default seam) -----------------------------------------------------


def test_now_returns_timezone_aware_datetime():
    n = datetime_tool._now()
    assert isinstance(n, datetime)
    # astimezone() makes it aware; a naive datetime (mutation dropping it) fails here.
    assert n.tzinfo is not None


# --- build_datetime_tool -----------------------------------------------------


def test_tool_is_named_get_current_datetime():
    tool = datetime_tool.build_datetime_tool(now=lambda: _FIXED)
    assert tool.name == datetime_tool.DATETIME_TOOL_NAME


def test_tool_returns_formatted_current_datetime():
    tool = datetime_tool.build_datetime_tool(now=lambda: _FIXED)
    assert tool.invoke({}) == _EXPECTED
