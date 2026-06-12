from __future__ import annotations

import enum

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import jsonshape, options, output, theme, timeparse
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


def _session_rows(value: object) -> list[dict[str, object]]:
    return jsonshape.mapping_list(value)


class SessionStatus(enum.StrEnum):
    """Closed value set for ``sessions list --status`` (the API's lifecycle states),
    so a typo is rejected with Typer's choices error instead of silently filtering
    nothing (mirrors the ``choices.TranscriptOutput`` pattern)."""

    created = "created"
    completed = "completed"
    error = "error"


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List recent streaming sessions", "assembly sessions list"),
            ("Find failed sessions", "assembly sessions list --status error"),
            (
                "Inspect the most recent session",
                "assembly sessions get $(assembly sessions list --json | jq -r '.[0].session_id')",
            ),
            (
                "Total audio across recent sessions (seconds)",
                "assembly sessions list --json | jq '[.[].audio_duration_sec] | add'",
            ),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(10, "--limit", help="How many sessions to show.", min=1),
    status: SessionStatus | None = typer.Option(
        None, "--status", help="Only show sessions with this status."
    ),
    json_out: bool = options.json_option(),
) -> None:
    """List recent streaming sessions."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_streaming(
            jwt, limit=limit, status=None if status is None else status.value
        )
        rows = _session_rows(payload.get("data"))

        def render(data: list[dict[str, object]]) -> object:
            if not data:
                return output.muted("No streaming sessions yet.")
            table = output.data_table(
                "session id",
                "status",
                "created (UTC)",
                "audio (s)",
                "model",
            )
            for s in data:
                table.add_row(
                    escape(str(s["session_id"])),
                    theme.status_text(str(s["status"])),
                    escape(timeparse.format_utc_datetime(s.get("created_at"))),
                    escape(str(s.get("audio_duration_sec") or "")),
                    escape(str(s.get("speech_model") or "")),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Show one session's details", "assembly sessions get sess_5551234"),
            ("Raw JSON for one session", "assembly sessions get sess_5551234 --json"),
            (
                "Drill into the latest session",
                "assembly sessions get $(assembly sessions list --json | jq -r '.[0].session_id')",
            ),
        ]
    )
)
def get(
    ctx: typer.Context,
    session_id: str = typer.Argument(..., help="Streaming session id."),
    json_out: bool = options.json_option(),
) -> None:
    """Show details for one streaming session."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        data = ams.get_streaming(session_id, jwt)

        def render(d: dict[str, object]) -> Table:
            table = output.detail_table()
            for field in _DETAIL_FIELDS:
                value = d.get(field)
                label = field.replace("_", " ")
                table.add_row(label, escape("" if value is None else str(value)))
            return table

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
