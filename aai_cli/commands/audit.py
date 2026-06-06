from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import typer
from rich.console import Group
from rich.markup import escape
from rich.text import Text

from aai_cli import help_panels, jsonshape, output, timeparse
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="View your account's audit log.")

_LOGIN_ACTIONS = {"login", "login.succeeded", "login__succeeded"}
_ACTION_LABELS = {
    "account.create": "Account created",
    "account.created": "Account created",
    "account__created": "Account created",
    "account.tos_accepted": "Terms accepted",
    "account.tos.accepted": "Terms accepted",
    "account__tos_accepted": "Terms accepted",
    "account.upgrade": "Account upgraded",
    "account.upgraded": "Account upgraded",
    "account__upgraded": "Account upgraded",
    "member.create": "Member created",
    "member.created": "Member created",
    "member__created": "Member created",
    "token.create": "API key created",
    "token.created": "API key created",
    "token__created": "API key created",
    "token.rename": "API key renamed",
    "token.renamed": "API key renamed",
    "token__renamed": "API key renamed",
    "login": "Login",
    "login.succeeded": "Login succeeded",
    "login__succeeded": "Login succeeded",
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


def _parse_time(value: object) -> datetime | None:
    parsed = timeparse.parse_iso_utc(value)
    return None if parsed is None else parsed.replace(tzinfo=None)


def _format_time(value: object) -> str:
    parsed = _parse_time(value)
    if parsed is None:
        return str(value or "")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


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
            ("Recent audit-log entries", "aai audit --limit 20"),
            ("Include login events", "aai audit --include-logins"),
            ("Filter by action", "aai audit --action token.create"),
        ]
    ),
)
def audit(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", help="How many entries to show."),
    action: str | None = typer.Option(None, "--action", help="Filter by raw action name."),
    resource: str | None = typer.Option(None, "--resource", help="Filter by raw resource type."),
    include_logins: bool = typer.Option(
        False, "--include-logins", help="Show successful login events."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List recent audit-log entries for your account."""

    def body(state: AppState, json_mode: bool) -> None:
        _, jwt = resolve_session(state)
        payload = ams.list_audit_logs(jwt, limit=limit, action_taken=action, resource_type=resource)
        rows = _audit_rows(payload)

        def render(data: list[dict[str, object]]) -> object:
            hide_logins = not include_logins and action is None
            shown = [entry for entry in data if not (hide_logins and _is_login(entry))]
            hidden_logins = len(data) - len(shown)
            if not shown:
                message = (
                    "No notable audit events in the recent log."
                    if hidden_logins
                    else "No audit events found."
                )
                if hidden_logins:
                    return Group(
                        Text(message, style="aai.muted"),
                        Text(
                            f"Hidden: {hidden_logins} login event(s). Use --include-logins to show them.",
                            style="aai.muted",
                        ),
                    )
                return Text(message, style="aai.muted")

            table = output.data_table("when (UTC)", "event", "resource", "actor")
            for entry in shown:
                table.add_row(
                    escape(_format_time(entry.get("log_time"))),
                    escape(_format_action(entry.get("action_taken"))),
                    escape(_resource_label(entry)),
                    escape(_actor_label(entry)),
                )
            if hidden_logins:
                return Group(
                    table,
                    Text(
                        f"Hidden: {hidden_logins} login event(s). Use --include-logins to show them.",
                        style="aai.muted",
                    ),
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
