from __future__ import annotations

from datetime import UTC, date, datetime, time


def parse_iso_utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = (
            datetime.fromisoformat(text)
            if "T" in text
            else datetime.combine(date.fromisoformat(text), time.min)
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
