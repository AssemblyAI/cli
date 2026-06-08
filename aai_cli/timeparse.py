from __future__ import annotations

from datetime import UTC, date, datetime, time


def _parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO date or datetime string to a (possibly naive) datetime.

    A bare date (no ``T``) becomes midnight; ``Z`` is accepted as the UTC suffix.
    Returns ``None`` when the string isn't a valid ISO date/datetime.
    """
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        if "T" in text:
            return datetime.fromisoformat(text)
        return datetime.combine(date.fromisoformat(text), time.min)
    except ValueError:
        return None


def parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = _parse_iso_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_utc_day(value: object) -> str:
    """Render an ISO timestamp as its UTC calendar day (``YYYY-MM-DD``).

    Falls back to the stringified input when it isn't a parseable timestamp, so a
    listing never blanks out a value it couldn't interpret.
    """
    parsed = parse_iso_utc(value)
    if parsed is None:
        return str(value or "")
    return parsed.date().isoformat()


def format_utc_datetime(value: object) -> str:
    """Render an ISO timestamp as a UTC ``YYYY-MM-DD HH:MM:SS`` string.

    Falls back to the stringified input when it isn't a parseable timestamp.
    """
    parsed = parse_iso_utc(value)
    if parsed is None:
        return str(value or "")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")
