from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import run_command
from aai_cli.commands.share import _exec as share_exec
from aai_cli.ui.help_text import examples_epilog

# Flattened single-command sub-typer (same pattern as `assembly dev`).
app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.BUILD,
    order=30,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("share",),
)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Share the running app on a public URL", "assembly share"),
            ("Use a specific local port", "assembly share --port 8000"),
            ("Skip the dependency install step", "assembly share --no-install"),
        ]
    ),
)
def share(
    ctx: typer.Context,
    port: int = typer.Option(3000, "--port", help="Local server port", min=0, max=65535),
    no_install: bool = typer.Option(
        False, "--no-install", help="Skip dependency install; launch directly"
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Expose the local app on a public URL via a cloudflared tunnel

    Run this from inside a project created by `assembly init`. It starts the dev server and
    opens a cloudflared quick tunnel, printing a shareable https://*.trycloudflare.com
    URL. Requires cloudflared (macOS: `brew install cloudflared`; other platforms:
    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
    """
    opts = share_exec.ShareOptions(port=port, no_install=no_install)
    run_command(
        ctx,
        lambda state, json_mode: share_exec.run_share(opts, state, json_mode=json_mode),
        json=json_out,
    )
