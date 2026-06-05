from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from aai_cli import output, theme
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="Browse your past streaming (real-time) sessions.", no_args_is_help=True)

# Fields shown by `sessions get`, in display order.
_DETAIL_FIELDS = (
    "session_id",
    "status",
    "region",
    "created_at",
    "completed_at",
    "audio_duration_sec",
    "session_duration_sec",
    "speech_model",
    "language_code",
    "error",
)


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List recent streaming sessions", "aai sessions list"),
            ("Only completed sessions", "aai sessions list --status completed"),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many sessions to show."),
    status: str = typer.Option(None, "--status", help="Filter: created, completed, or error."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent streaming sessions."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_streaming(jwt, limit=limit, status=status)
        rows = payload.get("data", [])

        def render(data: list[dict]) -> Table:
            table = Table(
                "session id",
                "status",
                "created",
                "audio (s)",
                "model",
                header_style="aai.heading",
            )
            for s in data:
                status_str = str(s["status"])
                table.add_row(
                    escape(str(s["session_id"])),
                    Text(status_str, style=theme.status_style(status_str)),
                    escape(str(s.get("created_at") or "")),
                    escape(str(s.get("audio_duration_sec") or "")),
                    escape(str(s.get("speech_model") or "")),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Show one session's details", "aai sessions get <session-id>"),
        ]
    )
)
def get(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Streaming session id."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Show details for one streaming session."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        data = ams.get_streaming(session_id, jwt)

        def render(d: dict) -> Table:
            table = Table(show_header=False)
            for field in _DETAIL_FIELDS:
                value = d.get(field)
                table.add_row(field, escape("" if value is None else str(value)))
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
