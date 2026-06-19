from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Annotated

import typer
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from rich.markup import escape
from rich.text import Text

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.auth import ams
from aai_cli.core import choices, jsonshape, timeparse
from aai_cli.core.errors import UsageError
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog


def _parse_day(day: str) -> date:
    try:
        return date.fromisoformat(day)
    except ValueError as exc:
        raise UsageError(f"Invalid date {day!r}; expected YYYY-MM-DD.") from exc


def _utc_day_start(day: date) -> str:
    """Render a date as a tz-aware UTC ISO-8601 timestamp.

    The AMS billing endpoint compares the bounds against tz-aware datetimes and
    rejects naive ones ("can't compare offset-naive and offset-aware datetimes"),
    so the wire value always carries an explicit ``+00:00`` offset.
    """
    return datetime(day.year, day.month, day.day, tzinfo=UTC).isoformat()


def _format_usage_number(value: object) -> str:
    number = jsonshape.as_float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def _format_dollars(cents: float) -> str:
    return f"${cents / 100:,.2f}"


# The product/feature label keys a usage line item may carry, in preference order.
_LABEL_KEYS = ("name", "product", "service", "feature", "model", "type", "description")

# AMS payload shapes drift, so the usage models are deliberately tolerant: unknown
# fields are ignored, a junk price falls back to 0.0, and a non-list/non-object
# never raises — `assembly usage` must degrade gracefully, not crash. They are
# parse-side only: `--json` passes the raw AMS dict through untouched.
_MappingList = BeforeValidator(jsonshape.mapping_list)


class _LineItem(BaseModel):
    """One usage line item: a ``price`` in cents plus whichever label key AMS used."""

    model_config = ConfigDict(extra="allow")

    price: Annotated[float, BeforeValidator(jsonshape.as_float)] = 0.0

    @property
    def label(self) -> str:
        """The product/feature label for the item, or ``""`` if it carries none."""
        extra = self.model_extra or {}
        return next((str(value) for key in _LABEL_KEYS if (value := extra.get(key))), "")


class _Window(BaseModel):
    """One usage window.

    The AMS usage endpoint returns ``total: 0.0`` on every window; the real spend
    lives in each window's ``line_items[].price`` (cents, like ``balance_in_cents``),
    so the window total is derived from them rather than the dead top-level ``total``.
    """

    start_timestamp: object = None
    end_timestamp: object = None
    line_items: Annotated[list[_LineItem], _MappingList] = Field(default_factory=list[_LineItem])

    @property
    def total_cents(self) -> float:
        return sum(item.price for item in self.line_items)

    @property
    def label(self) -> str:
        start = timeparse.parse_iso_utc(self.start_timestamp)
        end = timeparse.parse_iso_utc(self.end_timestamp)
        if start is None or end is None:
            return timeparse.format_utc_day(self.start_timestamp)
        if end.date() == start.date() + timedelta(days=1):
            return start.date().isoformat()
        return f"{start.date().isoformat()} to {end.date().isoformat()}"

    @property
    def breakdown(self) -> str:
        """Per-product spend for the window, in dollars, aggregated by product and
        ordered biggest-first.

        Both this and ``total_cents`` derive from ``line_items[].price`` (cents), so
        the breakdown is shown in the same unit as the ``total`` column and the
        products sum to that total — they reconcile, instead of mixing dollars with
        raw quantities. Products are aggregated by label (the AMS endpoint can return
        several rows for one product), a row with no recognizable product is grouped
        under ``other``, and zero-dollar products are dropped as noise (they don't
        affect the reconciliation).
        """
        totals: dict[str, float] = {}
        for item in self.line_items:
            name = item.label or "other"
            totals[name] = totals.get(name, 0.0) + item.price
        ordered = sorted(((n, c) for n, c in totals.items() if c), key=lambda nc: (-nc[1], nc[0]))
        return ", ".join(f"{name}: {_format_dollars(cents)}" for name, cents in ordered)


class _Usage(BaseModel):
    """The AMS usage response: just the windows; everything else is passthrough."""

    usage_items: Annotated[list[_Window], _MappingList] = Field(default_factory=list[_Window])


app = typer.Typer(help="Account billing, usage, and limits")

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.ACCOUNT,
    order=20,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("balance", "usage", "limits"),
)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show your remaining balance", "assembly balance"),
            ("Get the raw cents for scripting", "assembly balance -o balance_in_cents"),
        ]
    ),
)
def balance(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
) -> None:
    """Show your remaining account balance"""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = state.resolve_session()
        data = ams.get_balance(jwt)
        cents = jsonshape.as_float(data.get("balance_in_cents"))
        output.emit(
            data,
            lambda _d: f"Balance: [aai.success]{_format_dollars(cents)}[/aai.success]",
            json_mode=json_mode,
            fields=fields,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Usage over the last 30 days", "assembly usage"),
            ("A specific date range", "assembly usage --start 2026-05-01 --end 2026-06-01"),
            ("Break spend down by month", "assembly usage --window month"),
            (
                "Total spend in cents for scripting",
                "assembly usage --json | jq '[.usage_items[].line_items[].price] | add'",
            ),
        ]
    ),
)
def usage(
    ctx: typer.Context,
    start: str | None = typer.Option(
        None, "--start", help="Start date (YYYY-MM-DD). Default: 30d ago."
    ),
    end: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD). Default: today."),
    window: choices.UsageWindow | None = typer.Option(
        None, "--window", help="Aggregate usage by this window size"
    ),
    include_zero: bool = typer.Option(
        False,
        "--include-zero",
        "--all",
        help="Include zero-usage windows (matches --include-logins on `assembly audit`)",
    ),
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
) -> None:
    """Show usage over a date range (default: last 30 days)"""

    def body(state: AppState, json_mode: bool) -> None:
        # Parse/validate the flags before any session resolution or network work,
        # so a bad --start/--end/--window is a fast usage error even when not logged in.
        today = datetime.now(UTC).date()
        start_day = _parse_day(start) if start else today - timedelta(days=30)
        end_day = _parse_day(end) if end else today
        if end_day < start_day:
            raise UsageError(
                f"--end {end_day.isoformat()} is before --start {start_day.isoformat()}.",
                suggestion="Pick an end date on or after the start date.",
            )
        start_date = _utc_day_start(start_day)
        end_date = _utc_day_start(end_day)
        _, jwt = state.resolve_session()
        data = ams.get_usage(jwt, start_date, end_date, window)

        def render(d: dict[str, object]) -> object:
            windows = [(item, item.total_cents) for item in _Usage.model_validate(d).usage_items]
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

            shown_with_breakdown = [(item, cents, item.breakdown) for item, cents in shown]
            show_breakdown = any(breakdown for _, _, breakdown in shown_with_breakdown)
            table = (
                output.data_table("period", "total", "breakdown")
                if show_breakdown
                else output.data_table("period", "total")
            )
            hidden_count = len(windows) - len(shown)
            for item, cents, breakdown in shown_with_breakdown:
                row = [escape(item.label), _format_dollars(cents)]
                if show_breakdown:
                    row.append(escape(breakdown))
                table.add_row(*row)
            hidden_note = output.hidden_note(hidden_count, "zero-usage window", "--include-zero")
            return output.stack(summary, table, hidden_note)

        output.emit(data, render, json_mode=json_mode, fields=fields)

    run_command(ctx, body, json=json_out)


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Show rate limits per service", "assembly limits"),
            ("As JSON for scripting", "assembly limits --json"),
        ]
    ),
)
def limits(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
) -> None:
    """Show your account's rate limits per service"""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = state.resolve_session()
        data = ams.get_rate_limits(account_id, jwt)

        def render(d: dict[str, object]) -> object:
            limits = jsonshape.mapping_list(d.get("rate_limits"))
            if not limits:
                return output.muted(
                    "No custom rate limits — this account uses AssemblyAI's standard limits."
                )
            table = output.data_table("service", "limit")
            for limit in limits:
                table.add_row(
                    escape(str(limit.get("service", ""))),
                    _format_usage_number(limit.get("magnitude")),
                )
            return table

        output.emit(data, render, json_mode=json_mode, fields=fields)

    run_command(ctx, body, json=json_out)
