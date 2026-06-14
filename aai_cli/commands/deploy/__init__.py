from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_with_options
from aai_cli.commands.deploy import _exec as deploy_exec
from aai_cli.ui.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `assembly dev`).
app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.BUILD,
    order=40,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("deploy",),
)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Deploy a preview to Vercel (asks first)", "assembly deploy"),
            ("Deploy to production on Vercel", "assembly deploy --prod --yes"),
            ("Deploy to Railway", "assembly deploy --railway"),
            ("Deploy to Fly.io", "assembly deploy --fly"),
        ]
    ),
)
def deploy(
    ctx: typer.Context,
    prod: bool = typer.Option(False, "--prod", help="Deploy to production (Vercel only)"),
    vercel: bool = typer.Option(False, "--vercel", help="Deploy to Vercel (the default)"),
    railway: bool = typer.Option(False, "--railway", help="Deploy to Railway"),
    fly: bool = typer.Option(False, "--fly", help="Deploy to Fly.io"),
    assume_yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt"),
    json_out: bool = options.json_option(),
) -> None:
    """Deploy the current project to Vercel, Railway, or Fly.io

    Asks for confirmation first, then runs the target's CLI (`vercel deploy`,
    `railway up`, or `fly launch`). Requires that target's CLI to be installed.
    (Render deploys from a connected Git repo — see the project README.)
    """
    opts = deploy_exec.DeployOptions(
        prod=prod, vercel=vercel, railway=railway, fly=fly, assume_yes=assume_yes
    )
    run_with_options(ctx, deploy_exec.run_deploy, opts, json=json_out)
