from __future__ import annotations

import typer
from rich.markup import escape
from rich.table import Table

from aai_cli import jsonshape, output
from aai_cli.auth import ams
from aai_cli.context import AppState, resolve_session, run_command
from aai_cli.errors import APIError
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="List, create, and rename your AssemblyAI API keys.", no_args_is_help=True)


def _mapping(value: object) -> dict[str, object] | None:
    return jsonshape.as_mapping(value)


def _mapping_list(value: object) -> list[dict[str, object]]:
    return jsonshape.mapping_list(value)


def _project_id(project: dict[str, object]) -> int | None:
    value = project.get("id")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List your API keys (masked)", "aai keys list"),
            ("As JSON for scripting", "aai keys list --json"),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List API keys across your projects (keys shown masked)."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        projects = ams.list_projects(account_id, jwt)
        rows: list[dict[str, object]] = []
        for entry in projects:
            project = _mapping(entry.get("project")) or {}
            project_name = project.get("name", "")
            rows.extend(
                {
                    "id": token.get("id", ""),
                    "name": token.get("name") or token.get("token_name", ""),
                    "project": project_name,
                    "key": output.mask_secret(str(token.get("api_key", ""))),
                    "disabled": bool(token.get("is_disabled")),
                }
                for token in _mapping_list(entry.get("tokens"))
            )

        def render(data: list[dict[str, object]]) -> Table:
            table = Table("id", "name", "project", "key", "disabled", header_style="aai.heading")
            for row in data:
                table.add_row(
                    str(row["id"]),
                    escape(str(row["name"])),
                    escape(str(row["project"])),
                    escape(str(row["key"])),
                    "yes" if row["disabled"] else "no",
                )
            return table

        output.emit(rows, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Create a key in your default project", "aai keys create --name ci-pipeline"),
            ("Create a key in a specific project", "aai keys create --name prod --project 7"),
        ]
    )
)
def create(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="A label for the new key."),
    project_id: int | None = typer.Option(
        None, "--project", help="Project id to create the key in (defaults to your first)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Create a new API key. Prints the key value once — copy it now."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        pid = project_id
        if pid is None:
            projects = ams.list_projects(account_id, jwt)
            if not projects:
                raise APIError("Your account has no project to create a key in.")
            project = _mapping(projects[0].get("project"))
            if project is None:
                raise APIError("Your account has no project to create a key in.")
            pid = _project_id(project)
            if pid is None:
                raise APIError("Your account has no project to create a key in.")
        created = ams.create_token(account_id, pid, name, jwt)
        output.emit(
            created,
            lambda d: (
                output.success(f"Created API key '{escape(name)}'.")
                + f"\n  {escape(str(d['api_key']))}\n"
                + output.warn("Shown once — copy it now.")
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Relabel a key (id from `aai keys list`)", 'aai keys rename 123 "prod"'),
        ]
    )
)
def rename(
    ctx: typer.Context,
    token_id: int = typer.Argument(..., help="The key id (see `aai keys list`)."),
    new_name: str = typer.Argument(..., help="The new label."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Rename an existing API key."""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = resolve_session(state)
        ams.rename_token(account_id, token_id, new_name, jwt)
        output.emit(
            {"id": token_id, "name": new_name},
            lambda d: f"Renamed key {d['id']} to '{escape(new_name)}'.",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
