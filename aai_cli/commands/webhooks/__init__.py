# aai_cli/commands/webhooks/__init__.py
from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.commands.webhooks import _listen as webhook_listen
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer(help="Receive webhook deliveries on a public dev URL", no_args_is_help=True)

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.TRANSCRIPTION,
    order=110,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("webhooks",),
    group_name="webhooks",
)


@app.command(
    epilog=examples_epilog(
        [
            ("Listen on a public URL", "assembly webhooks listen"),
            (
                "Deliver a transcription to it",
                "assembly transcribe call.mp3 --webhook-url https://….trycloudflare.com",
            ),
            (
                "Forward deliveries to your app",
                "assembly webhooks listen --forward-to http://localhost:8000/webhook",
            ),
            ("Local-only sink (no tunnel)", "assembly webhooks listen --no-tunnel"),
            ("Stop after the first delivery", "assembly webhooks listen --max-events 1"),
        ]
    ),
)
def listen(
    ctx: typer.Context,
    port: int = typer.Option(
        8989, "--port", help="Local listener port (the first free port from here)"
    ),
    forward_to: str | None = typer.Option(
        None, "--forward-to", help="Re-POST each delivery to this URL (e.g. your local app)"
    ),
    no_tunnel: bool = typer.Option(
        False, "--no-tunnel", help="Local-only: skip the cloudflared public URL"
    ),
    max_events: int = typer.Option(
        0,  # pragma: no mutate (0 = serve forever; only observable by never exiting)
        "--max-events",
        min=0,
        help="Exit after this many deliveries (0 = run until Ctrl-C)",
    ),
    json_out: bool = options.json_option("One NDJSON record per delivery"),
) -> None:
    """Receive AssemblyAI webhooks on a public URL and watch them arrive

    Opens a cloudflared quick tunnel to a local listener and prints the public
    URL to pass as --webhook-url (or webhook_url in the API). Each delivery is
    acknowledged with HTTP 200 and printed; --forward-to relays it to your app,
    so you can build webhook handlers without deploying a public endpoint.

    Requires cloudflared unless --no-tunnel (macOS: `brew install cloudflared`; other
    platforms: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
    """

    def body(_state: AppState, json_mode: bool) -> None:
        webhook_listen.run_listen(
            webhook_listen.ListenOptions(
                port=port,
                forward_to=forward_to,
                public=not no_tunnel,
                max_events=max_events,
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
