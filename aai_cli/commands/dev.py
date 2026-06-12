# aai_cli/commands/dev.py
from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import help_panels, options, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.init import devserver, procfile, runner

# Flattened single-command sub-typer (same pattern as `assembly init`): one
# @app.command() registered via app.add_typer(dev.app) with no name.
app = typer.Typer()


def run_dev(
    *, port: int, host: str, no_install: bool, no_open: bool, json_mode: bool, quiet: bool
) -> None:
    """Boot the project's Procfile `web:` process locally, with live reload."""
    target = Path.cwd()
    use_uv = runner.has_uv()

    chosen_port = runner.find_free_port(port)
    devserver.notify_port_change(port, chosen_port, json_mode=json_mode, quiet=quiet)
    env = {**os.environ, "PORT": str(chosen_port)}
    # Resolves the start command AND validates we're inside a scaffolded project.
    web = procfile.web_argv(target, env=env)

    report: list[steps.Step] = [
        devserver.install_step(target, no_install=no_install, use_uv=use_uv)
    ]
    output.emit(report, lambda d: steps.render_steps(d, heading="Dev"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    command = devserver.dev_command(target, web, use_uv=use_uv, host=host)
    # The printed URL reflects the actual bind: "localhost" for the loopback
    # default, the literal host for an explicit --host.
    url_host = "localhost" if host == devserver.LOCAL_HOST else host
    url = f"http://{url_host}:{chosen_port}"
    if not json_mode:
        output.console.print(
            f"[aai.heading]Starting[/aai.heading] [aai.url]{escape(url)}[/aai.url]"
            "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
        )
    code = runner.run_server(
        target, command=command, port=chosen_port, env=env, open_browser=not no_open
    )
    if code:
        raise typer.Exit(code=code)


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
    port: int = typer.Option(3000, "--port", help="Local server port."),
    host: str = typer.Option(
        devserver.LOCAL_HOST,
        "--host",
        help="Interface to bind. Loopback by default; pass 0.0.0.0 to expose on your network.",
    ),
    no_open: bool = typer.Option(False, "--no-open", help="Launch, but don't open the browser."),
    no_install: bool = typer.Option(
        False, "--no-install", help="Skip dependency install; launch directly."
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Launch the dev server for the app in the current directory.

    Run this from inside a project created by `assembly init`. It installs dependencies
    if needed, then starts the FastAPI server with live reload and opens the browser.
    """

    def body(state: AppState, json_mode: bool) -> None:
        run_dev(
            port=port,
            host=host,
            no_install=no_install,
            no_open=no_open,
            json_mode=json_mode,
            quiet=state.quiet,
        )

    run_command(ctx, body, json=json_out)
