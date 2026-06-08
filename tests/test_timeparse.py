from datetime import UTC

from aai_cli import timeparse


def test_parse_iso_utc_normalizes_z_and_offsets():
    parsed = timeparse.parse_iso_utc("2026-06-01T12:00:00-04:00")
    assert parsed is not None
    assert parsed.tzinfo is UTC
    assert parsed.isoformat() == "2026-06-01T16:00:00+00:00"

    zulu = timeparse.parse_iso_utc("2026-06-01T12:00:00Z")
    assert zulu is not None
    assert zulu.isoformat() == "2026-06-01T12:00:00+00:00"


def test_parse_iso_utc_treats_dates_and_naive_datetimes_as_utc():
    date_only = timeparse.parse_iso_utc("2026-06-01")
    assert date_only is not None
    assert date_only.isoformat() == "2026-06-01T00:00:00+00:00"

    naive = timeparse.parse_iso_utc("2026-06-01T12:00:00")
    assert naive is not None
    assert naive.isoformat() == "2026-06-01T12:00:00+00:00"


def test_parse_iso_utc_rejects_non_dates():
    assert timeparse.parse_iso_utc(None) is None
    assert timeparse.parse_iso_utc("") is None
    assert timeparse.parse_iso_utc("not-a-date") is None
    # A truthy non-string must also be rejected (not just falsy None/""). This pins
    # the `not isinstance(...) or not value` guard: an `and` there would fall through
    # to str-only operations on the int and raise instead of returning None.
    assert timeparse.parse_iso_utc(20260601) is None
    assert timeparse.parse_iso_utc(["2026-06-01"]) is None


def test_format_utc_day_renders_calendar_day_or_falls_back():
    assert timeparse.format_utc_day("2026-06-01T12:00:00-04:00") == "2026-06-01"
    assert timeparse.format_utc_day("2026-06-01") == "2026-06-01"
    # Unparseable input falls back to the stringified value, and None to "".
    assert timeparse.format_utc_day("bad") == "bad"
    assert timeparse.format_utc_day(None) == ""


def test_format_utc_datetime_renders_seconds_or_falls_back():
    assert timeparse.format_utc_datetime("2026-06-01T12:00:00") == "2026-06-01 12:00:00"
    # Offsets are normalized to UTC before formatting.
    assert timeparse.format_utc_datetime("2026-06-01T12:00:00-04:00") == "2026-06-01 16:00:00"
    assert timeparse.format_utc_datetime("bad") == "bad"
    assert timeparse.format_utc_datetime(None) == ""
