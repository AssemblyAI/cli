from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import output
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="View your account's audit log.")


@app.command(
    epilog=examples_epilog(
        [
            ("Recent audit-log entries", "aai audit --limit 20"),
            ("Filter by action", "aai audit --action token.create"),
        ]
    )
)
def audit(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="How many entries to show."),
    action: str = typer.Option(None, "--action", help="Filter by action_taken."),
    resource: str = typer.Option(None, "--resource", help="Filter by resource_type."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent audit-log entries for your account."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_audit_logs(jwt, limit=limit, action_taken=action, resource_type=resource)
        rows = payload.get("data", [])

        def render(data: list[dict[str, object]]) -> Table:
            table = Table("time", "actor", "action", "resource", header_style="aai.heading")
            for entry in data:
                actor = str(entry["actor_type"])
                actor_id = entry.get("actor_id")
                if actor_id is not None:
                    actor = f"{actor}:{actor_id}"
                resource_label = entry.get("resource_type") or ""
                resource_id = entry.get("resource_id")
                if resource_label and resource_id:
                    resource_label = f"{resource_label}:{resource_id}"
                table.add_row(
                    escape(str(entry["log_time"])),
                    escape(actor),
                    escape(str(entry["action_taken"])),
                    escape(str(resource_label)),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
