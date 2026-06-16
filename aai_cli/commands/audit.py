from __future__ import annotations

from collections.abc import Mapping

import typer
from rich.markup import escape

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.auth import ams
from aai_cli.core import jsonshape, timeparse
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer(help="View your account's audit log")

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.ACCOUNT,
    order=40,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("audit",),
)

# `__`-separated variants are handled by _normalize_action (which maps `__` -> `.`),
# so only the dotted forms need entries here.
_LOGIN_ACTIONS = {"login", "login.succeeded"}
_ACTION_LABELS = {
    "account.create": "Account created",
    "account.created": "Account created",
    "account.tos_accepted": "Terms accepted",
    "account.tos.accepted": "Terms accepted",
    "account.upgrade": "Account upgraded",
    "account.upgraded": "Account upgraded",
    "member.create": "Member created",
    "member.created": "Member created",
    "token.create": "API key created",
    "token.created": "API key created",
    "token.rename": "API key renamed",
    "token.renamed": "API key renamed",
    "login": "Login",
    "login.succeeded": "Login succeeded",
}


def _normalize_action(action: object) -> str:
    return str(action or "").replace("__", ".")


def _format_action(action: object) -> str:
    raw = str(action or "")
    if raw in _ACTION_LABELS:
        return _ACTION_LABELS[raw]
    normalized = _normalize_action(raw)
    if normalized in _ACTION_LABELS:
        return _ACTION_LABELS[normalized]
    return normalized.replace("_", " ").replace(".", " ").strip().capitalize() or "Unknown"


def _is_login(entry: Mapping[str, object]) -> bool:
    raw = str(entry.get("action_taken") or "")
    return raw in _LOGIN_ACTIONS or _normalize_action(raw) in _LOGIN_ACTIONS


def _actor_label(entry: Mapping[str, object]) -> str:
    actor_type = str(entry.get("actor_type") or "system")
    actor_id = entry.get("actor_id")
    if actor_type == "system":
        return "system" if actor_id is None else f"system #{actor_id}"
    return actor_type if actor_id is None else f"{actor_type} #{actor_id}"


def _resource_label(entry: Mapping[str, object]) -> str:
    resource_type = entry.get("resource_type")
    if not resource_type:
        return ""
    resource_id = entry.get("resource_id")
    label = str(resource_type).replace("_", " ")
    return label if not resource_id else f"{label} #{resource_id}"


def _audit_rows(payload: Mapping[str, object]) -> list[dict[str, object]]:
    return jsonshape.mapping_list(payload.get("data"))


@app.command(
    rich_help_panel=help_panels.ACCOUNT,
    epilog=examples_epilog(
        [
            ("Recent audit-log entries", "assembly audit"),
            ("Show more entries", "assembly audit --limit 100"),
            ("Include login events", "assembly audit --include-logins"),
            ("Filter by action", "assembly audit --action token.create"),
            ("Filter by resource, as JSON", "assembly audit --resource token --json"),
            ("Pull action and actor as columns", "assembly audit -o action_taken,actor_id"),
        ]
    ),
)
def audit(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", min=1, help="How many entries to show"),
    action: str | None = typer.Option(None, "--action", help="Filter by raw action name"),
    resource: str | None = typer.Option(None, "--resource", help="Filter by raw resource type"),
    include_logins: bool = typer.Option(
        False, "--include-logins", help="Show successful login events"
    ),
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
) -> None:
    """List recent audit-log entries for your account"""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = state.resolve_session()
        payload = ams.list_audit_logs(jwt, limit=limit, action_taken=action, resource_type=resource)
        rows = _audit_rows(payload)

        def render(data: list[dict[str, object]]) -> object:
            hide_logins = not include_logins and action is None
            shown = [entry for entry in data if not (hide_logins and _is_login(entry))]
            hidden_logins = len(data) - len(shown)
            hidden_note = output.hidden_note(hidden_logins, "login event", "--include-logins")
            if not shown:
                message = (
                    "No notable audit events in the recent log."
                    if hidden_logins
                    else "No audit events found."
                )
                return output.stack(output.muted(message), hidden_note)

            table = output.data_table("when (UTC)", "event", "resource", "actor")
            for entry in shown:
                table.add_row(
                    escape(timeparse.format_utc_datetime(entry.get("log_time"))),
                    escape(_format_action(entry.get("action_taken"))),
                    escape(_resource_label(entry)),
                    escape(_actor_label(entry)),
                )
            return output.stack(table, hidden_note)

        output.emit(rows, render, json_mode=json_mode, fields=fields)

    run_command(ctx, body, json=json_out)
