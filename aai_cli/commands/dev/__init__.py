# aai_cli/commands/dev/__init__.py
from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_command
from aai_cli.commands.dev import _exec as dev_exec
from aai_cli.init import devserver
from aai_cli.ui.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `assembly init`): one
# @app.command() registered via app.add_typer(dev.app) with no name.
app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.BUILD,
    order=20,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("dev",),
)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Launch the app in the current directory", "assembly dev"),
            ("Use a specific port", "assembly dev --port 8000"),
            ("Launch without opening a browser", "assembly dev --no-open"),
            ("Skip the dependency install step", "assembly dev --no-install"),
        ]
    ),
)
def dev(
    ctx: typer.Context,
    port: int = typer.Option(3000, "--port", help="Local server port"),
    host: str = typer.Option(
        devserver.LOCAL_HOST,
        "--host",
        help="Interface to bind. Loopback by default; pass 0.0.0.0 to expose on your network.",
    ),
    no_open: bool = typer.Option(False, "--no-open", help="Launch, but don't open the browser"),
    no_install: bool = typer.Option(
        False, "--no-install", help="Skip dependency install; launch directly"
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Run the dev server for the app in the current directory

    Run this from inside a project created by `assembly init`. It installs dependencies
    if needed, then starts the FastAPI server with live reload and opens the browser.
    """
    opts = dev_exec.DevOptions(port=port, host=host, no_install=no_install, no_open=no_open)
    run_command(
        ctx,
        lambda state, json_mode: dev_exec.run_dev(opts, state, json_mode=json_mode),
        json=json_out,
    )
