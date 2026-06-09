from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta

import typer
from rich.markup import escape
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


def _format_usage_number(value: object) -> str:
    number = jsonshape.as_float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _usage_items(data: Mapping[str, object]) -> list[dict[str, object]]:
    return jsonshape.mapping_list(data.get("usage_items"))


def _format_dollars(cents: float) -> str:
    return f"${cents / 100:,.2f}"


def _window_total_cents(item: Mapping[str, object]) -> float:
    """Sum a window's spend (cents) from its ``line_items``.

    The AMS usage endpoint returns ``total: 0.0`` on every window; the real
    spend lives in each window's ``line_items[].price`` (cents, like
    ``balance_in_cents``), so the window total is derived from them rather than
    the dead top-level ``total``.
    """
    return sum(
        jsonshape.as_float(line_item.get("price"))
        for line_item in jsonshape.mapping_list(item.get("line_items"))
    )


def _window_label(item: Mapping[str, object]) -> str:
    start = timeparse.parse_iso_utc(item.get("start_timestamp"))
    end = timeparse.parse_iso_utc(item.get("end_timestamp"))
    if start is None or end is None:
        return timeparse.format_utc_day(item.get("start_timestamp"))
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
        for line_item in jsonshape.mapping_list(item.get("line_items"))
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
        cents = jsonshape.as_float(data.get("balance_in_cents"))
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
            windows = [(item, _window_total_cents(item)) for item in _usage_items(d)]
            shown = windows if include_zero else [w for w in windows if w[1]]
            total = sum(cents for _, cents in windows)
            range_label = (
                f"{timeparse.format_utc_day(start_date)} to "
                f"{timeparse.format_utc_day(end_date)} (UTC)"
            )
            summary = Text(
                f"Usage total: {_format_dollars(total)} for {range_label}",
                style="aai.heading",
            )
            if not shown:
                message = (
                    "No usage in this range."
                    if windows
                    else "No usage windows returned for this range."
                )
                return output.stack(summary, output.muted(message))

            shown_with_breakdown = [
                (item, cents, _line_items_summary(item)) for item, cents in shown
            ]
            show_breakdown = any(breakdown for _, _, breakdown in shown_with_breakdown)
            table = (
                output.data_table("period", "total", "breakdown")
                if show_breakdown
                else output.data_table("period", "total")
            )
            hidden_count = len(windows) - len(shown)
            for item, cents, breakdown in shown_with_breakdown:
                row = [
                    escape(_window_label(item)),
                    _format_dollars(cents),
                ]
                if show_breakdown:
                    row.append(escape(breakdown))
                table.add_row(*row)
            hidden_note = (
                output.muted(
                    f"Hidden: {hidden_count} zero-usage window(s). Use --all to show them."
                )
                if hidden_count
                else None
            )
            return output.stack(summary, table, hidden_note)

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

        def render(d: dict[str, object]) -> object:
            limits = jsonshape.mapping_list(d.get("rate_limits"))
            if not limits:
                return output.muted("No custom rate limits configured for this account.")
            table = output.data_table("service", "limit")
            for limit in limits:
                table.add_row(
                    escape(str(limit.get("service", ""))),
                    _format_usage_number(limit.get("magnitude")),
                )
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
