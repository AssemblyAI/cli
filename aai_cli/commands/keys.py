from __future__ import annotations

import typer
from rich.markup import escape

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.auth import ams
from aai_cli.core import jsonshape
from aai_cli.core.errors import APIError, UsageError
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer(help="List, create, and rename your AssemblyAI API keys", no_args_is_help=True)

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.ACCOUNT,
    order=30,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("keys",),
    group_name="keys",
)


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


def _default_project_id(account_id: int, jwt: str) -> int:
    """The id of the account's first project, for `keys create` without ``--project``.

    No projects, an unparseable first entry, and a missing/invalid id are the same
    failure — there's nothing to create a key in — so they share one error.
    """
    projects = ams.list_projects(account_id, jwt)
    project = jsonshape.as_mapping(projects[0].get("project")) if projects else None
    pid = _project_id(project) if project is not None else None
    if pid is None:
        raise APIError(
            "Your account has no project to create a key in.",
            suggestion="Create a project in the AssemblyAI dashboard, then try again.",
        )
    return pid


@app.command(
    name="list",
    epilog=examples_epilog(
        [
            ("List your API keys (masked)", "assembly keys list"),
            ("As JSON for scripting", "assembly keys list --json"),
            ("Get key ids to use with rename", "assembly keys list -o id"),
        ]
    ),
)
def list_(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
    fields: str | None = options.fields_option(),
) -> None:
    """List API keys across your projects (shown masked)"""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = state.resolve_session()
        projects = ams.list_projects(account_id, jwt)
        rows: list[dict[str, object]] = []
        for entry in projects:
            project = jsonshape.as_mapping(entry.get("project")) or {}
            project_name = project.get("name", "")
            rows.extend(
                {
                    "id": token.get("id", ""),
                    "name": token.get("name") or token.get("token_name", ""),
                    "project": project_name,
                    "key": output.redact_secret(str(token.get("api_key", ""))),
                    "disabled": bool(token.get("is_disabled")),
                }
                for token in jsonshape.mapping_list(entry.get("tokens"))
            )

        def render(data: list[dict[str, object]]) -> object:
            if not data:
                return output.muted("No API keys found.")
            table = output.data_table("id", "name", "project", "key", "disabled")
            for row in data:
                table.add_row(
                    str(row["id"]),
                    escape(str(row["name"])),
                    escape(str(row["project"])),
                    escape(str(row["key"])),
                    "yes" if row["disabled"] else "no",
                )
            return table

        output.emit(rows, render, json_mode=json_mode, fields=fields)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Create a key in your default project", "assembly keys create --name ci-pipeline"),
            ("Create a key in a specific project", "assembly keys create --name prod --project 7"),
            (
                "Capture the new key into an env var",
                "export ASSEMBLYAI_API_KEY=$(assembly keys create --name ci --json | jq -r '.api_key')",
            ),
        ]
    )
)
def create(
    ctx: typer.Context,
    name: str = typer.Option(..., "--name", help="A label for the new key"),
    project_id: int | None = typer.Option(
        None, "--project", help="Project id to create the key in (defaults to your first)"
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Create an API key (printed once — copy it now)"""

    def body(state: AppState, json_mode: bool) -> None:
        # Validate locally before any auth/network work: an empty or whitespace-only
        # label would otherwise cost a session resolution plus an AMS round-trip.
        if not name.strip():
            raise UsageError(
                "--name must not be empty.",
                suggestion="Pass a label for the key, e.g. --name ci-pipeline.",
            )
        account_id, jwt = state.resolve_session()
        pid = project_id if project_id is not None else _default_project_id(account_id, jwt)
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
            ("Relabel a key (id from `assembly keys list`)", 'assembly keys rename 123 "prod"'),
        ]
    )
)
def rename(
    ctx: typer.Context,
    token_id: int = typer.Argument(..., help="The key id (see `assembly keys list`)"),
    new_name: str = typer.Argument(..., help="The new label"),
    json_out: bool = options.json_option(),
) -> None:
    """Rename an existing API key"""

    def body(state: AppState, json_mode: bool) -> None:
        account_id, jwt = state.resolve_session()
        ams.rename_token(account_id, token_id, new_name, jwt)
        output.emit(
            {"id": token_id, "name": new_name},
            lambda d: f"Renamed key {d['id']} to '{escape(new_name)}'.",
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
