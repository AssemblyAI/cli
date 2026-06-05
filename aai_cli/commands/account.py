from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta

import typer
from rich.console import Group
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from aai_cli import help_panels, jsonshape, output, timeparse
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.errors import UsageError
from aai_cli.help_text import examples_epilog


def _utc_day_start(day: str) -> str:
    """Render a ``YYYY-MM-DD`` date as a tz-aware UTC ISO-8601 timestamp.

    The AMS billing endpoint compares the bounds against tz-aware datetimes and
    rejects naive ones ("can't compare offset-naive and offset-aware datetimes"),
    so the wire value always carries an explicit ``+00:00`` offset.
    """
    try:
        parsed = date.fromisoformat(day)
    except ValueError as exc:
        raise UsageError(f"Invalid date {day!r}; expected YYYY-MM-DD.") from exc
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC).isoformat()


def _parse_usage_timestamp(value: object) -> datetime | None:
    return timeparse.parse_iso_utc(value)


def _format_usage_day(value: object) -> str:
    parsed = _parse_usage_timestamp(value)
    if parsed is None:
        return str(value or "")
    return parsed.date().isoformat()


def _usage_number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _format_usage_number(value: object) -> str:
    number = _usage_number(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _mapping_list(value: object) -> list[dict[str, object]]:
    return jsonshape.mapping_list(value)


def _usage_items(data: Mapping[str, object]) -> list[dict[str, object]]:
    return _mapping_list(data.get("usage_items"))


def _window_label(item: Mapping[str, object]) -> str:
    start = _parse_usage_timestamp(item.get("start_timestamp"))
    end = _parse_usage_timestamp(item.get("end_timestamp"))
    if start is None or end is None:
        return _format_usage_day(item.get("start_timestamp"))
    if end.date() == start.date() + timedelta(days=1):
        return start.date().isoformat()
    return f"{start.date().isoformat()} to {end.date().isoformat()}"


def _line_item_label(line_item: Mapping[str, object]) -> str:
    label = next(
        (
            str(value)
            for key in ("name", "product", "service", "feature", "model", "type", "description")
            if (value := line_item.get(key))
        ),
        "",
    )
    value = next(
        (
            line_item[key]
            for key in ("total", "quantity", "amount", "usage", "count")
            if key in line_item
        ),
        None,
    )
    if label and value is not None:
        return f"{label}: {_format_usage_number(value)}"
    if label:
        return label
    if value is not None:
        return _format_usage_number(value)
    return ""


def _line_items_summary(item: Mapping[str, object]) -> str:
    labels = [
        label
        for line_item in _mapping_list(item.get("line_items"))
        if (label := _line_item_label(line_item))
    ]
    return ", ".join(labels)


app = typer.Typer(help="Account billing, usage, and limits.")


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show your remaining balance", "aai balance"),
        ]
    ),
)
def balance(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show your remaining account balance."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        data = ams.get_balance(jwt)
        cents = _usage_number(data.get("balance_in_cents"))
        output.emit(
            data,
            lambda _d: f"Balance: [aai.success]${cents / 100:,.2f}[/aai.success]",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Usage over the last 30 days", "aai usage"),
            ("A specific date range", "aai usage --start 2026-05-01 --end 2026-06-01"),
        ]
    ),
)
def usage(
    ctx: typer.Context,
    start: str | None = typer.Option(
        None, "--start", help="Start date (YYYY-MM-DD). Default: 30d ago."
    ),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD). Default: today."),
    window: str | None = typer.Option(None, "--window", help="Window size, e.g. 'day' or 'month'."),
    include_zero: bool = typer.Option(False, "--all", help="Include zero-usage windows."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show usage over a date range (defaults to the last 30 days)."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        today = datetime.now(UTC).date()
        start_date = _utc_day_start(start or (today - timedelta(days=30)).isoformat())
        end_date = _utc_day_start(end or today.isoformat())
        data = ams.get_usage(jwt, start_date, end_date, window)

        def render(d: dict[str, object]) -> object:
            items = _usage_items(d)
            shown = (
                items
                if include_zero
                else [item for item in items if _usage_number(item.get("total"))]
            )
            total = sum(_usage_number(item.get("total")) for item in items)
            range_label = f"{_format_usage_day(start_date)} to {_format_usage_day(end_date)} (UTC)"
            summary = Text(
                f"Usage total: {_format_usage_number(total)} for {range_label}",
                style="aai.heading",
            )
            if not shown:
                if items:
                    return Group(
                        summary,
                        Text("No usage in this range.", style="aai.muted"),
                    )
                return Group(
                    summary,
                    Text("No usage windows returned for this range.", style="aai.muted"),
                )

            shown_with_breakdown = [(item, _line_items_summary(item)) for item in shown]
            show_breakdown = any(summary for _, summary in shown_with_breakdown)
            table = (
                Table("period", "total", "breakdown", header_style="aai.heading")
                if show_breakdown
                else Table("period", "total", header_style="aai.heading")
            )
            hidden_count = len(items) - len(shown)
            for item, breakdown in shown_with_breakdown:
                row = [
                    escape(_window_label(item)),
                    _format_usage_number(item.get("total")),
                ]
                if show_breakdown:
                    row.append(escape(breakdown))
                table.add_row(*row)
            if hidden_count:
                return Group(
                    summary,
                    table,
                    Text(
                        f"Hidden: {hidden_count} zero-usage window(s). Use --all to show them.",
                        style="aai.muted",
                    ),
                )
            return Group(summary, table)

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show rate limits per service", "aai limits"),
        ]
    ),
)
def limits(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show your account's rate limits per service."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        data = ams.get_rate_limits(account_id, jwt)

        def render(d: dict[str, object]) -> Table:
            table = Table("service", "limit", header_style="aai.heading")
            for limit in _mapping_list(d.get("rate_limits")):
                table.add_row(
                    escape(str(limit.get("service", ""))),
                    _format_usage_number(limit.get("magnitude")),
                )
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
