from __future__ import annotations

import typer

from aai_cli import choices, command_registry, help_panels, options, output, setup_exec
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer(
    help="Set up your coding agent for AssemblyAI (docs MCP + skills)",
    no_args_is_help=True,
)

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.SETUP,
    order=20,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("setup",),
    group_name="setup",
)


@app.command(
    epilog=examples_epilog(
        [
            ("Set up your coding agent for AssemblyAI", "assembly setup install"),
            ("Install for the current project only", "assembly setup install --scope project"),
            ("Reinstall everything even if already present", "assembly setup install --force"),
        ]
    )
)
def install(
    ctx: typer.Context,
    scope: choices.Scope = typer.Option(
        choices.Scope.user,
        "--scope",
        help="Config scope to register the MCP under. Presence is detected across all scopes.",
    ),
    force: bool = typer.Option(False, "--force", help="Reinstall even if already present"),
    json_out: bool = options.json_option(),
) -> None:
    """Set up your coding agent for AssemblyAI (docs MCP + skills)

    Installs three things: the assemblyai-docs MCP server (live API docs, via
    `claude mcp add`), the AssemblyAI skill (via `npx skills add`), and the bundled
    aai-cli skill (copied from this package, no network). Each step is idempotent
    and skipped if already present unless --force.
    """

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [
            setup_exec.install_mcp(scope, force=force),
            setup_exec.install_skill(force=force),
            setup_exec.install_cli_skill(force=force),
        ]
        output.emit({"steps": steps}, setup_exec.render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Show what's set up", "assembly setup status"),
            ("Print status as JSON", "assembly setup status --json"),
        ]
    )
)
def status(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Show whether the AssemblyAI MCP and skills are set up in your coding agent"""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [setup_exec.mcp_status(), setup_exec.skill_status(), setup_exec.cli_skill_status()]
        output.emit({"steps": steps}, setup_exec.render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog(
        [
            ("Remove the AssemblyAI MCP server and skills", "assembly setup remove"),
            ("Remove only from the project scope", "assembly setup remove --scope project"),
        ]
    )
)
def remove(
    ctx: typer.Context,
    scope: choices.Scope | None = typer.Option(
        None,
        "--scope",
        help=(
            "Only remove the MCP from this scope. "
            "Default: remove from whichever scope it exists in."
        ),
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Remove the AssemblyAI MCP and skills from your coding agent"""

    def body(_state: AppState, json_mode: bool) -> None:
        steps = [
            setup_exec.remove_mcp(scope),
            setup_exec.remove_skill(),
            setup_exec.remove_cli_skill(),
        ]
        output.emit({"steps": steps}, setup_exec.render, json_mode=json_mode)
        if any(s["status"] == "failed" for s in steps):
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
