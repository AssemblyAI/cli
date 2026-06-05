from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import help_panels, output
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
        cents = data.get("balance_in_cents", 0) or 0
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
    start: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD). Default: 30d ago."),
    end: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD). Default: today."),
    window: str = typer.Option(None, "--window", help="Window size, e.g. 'day' or 'month'."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show usage over a date range (defaults to the last 30 days)."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        today = datetime.now(UTC).date()
        start_date = _utc_day_start(start or (today - timedelta(days=30)).isoformat())
        end_date = _utc_day_start(end or today.isoformat())
        data = ams.get_usage(jwt, start_date, end_date, window)

        def render(d: dict[str, Any]) -> Table:
            table = Table("window start", "window end", "total", header_style="aai.heading")
            for item in d.get("usage_items", []):
                table.add_row(
                    escape(str(item["start_timestamp"])),
                    escape(str(item["end_timestamp"])),
                    f"{item['total']:,}",
                )
            return table

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

        def render(d: dict[str, Any]) -> Table:
            table = Table("service", "limit", header_style="aai.heading")
            for limit in d.get("rate_limits", []):
                table.add_row(escape(str(limit["service"])), f"{limit['magnitude']:,}")
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
