# aai_cli/commands/dev.py
from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import help_panels, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.init import procfile, runner

# Flattened single-command sub-typer (same pattern as `aai init`): one
# @app.command() registered via app.add_typer(dev.app) with no name.
app = typer.Typer()


def _install_step(target: Path, *, no_install: bool, use_uv: bool) -> steps.Step:
    """Install deps (unless --no-install) and return the report row."""
    if no_install:
        return {"name": "install", "status": "skipped", "detail": "--no-install"}
    setup = runner.run_setup(target, use_uv=use_uv)
    if setup.returncode != 0:
        return {
            "name": "install",
            "status": "failed",
            "detail": (setup.stderr or setup.stdout).strip()[:300],
        }
    return {"name": "install", "status": "installed", "detail": "uv" if use_uv else "venv + pip"}


def _dev_command(target: Path, web: list[str], *, use_uv: bool) -> list[str]:
    """The Procfile web process, run in the project venv with live reload.

    In the no-uv branch `web[0]` must be a `python -m`-runnable module; every current
    template's `web:` line starts with `uvicorn`.
    """
    prefix = ["uv", "run"] if use_uv else [str(runner.venv_python(target)), "-m"]
    return [*prefix, *web, "--reload"]


def run_dev(*, port: int, no_install: bool, no_open: bool, json_mode: bool) -> None:
    """Boot the project's Procfile `web:` process locally, with live reload."""
    target = Path.cwd()
    use_uv = runner.has_uv()

    chosen_port = runner.find_free_port(port)
    env = {**os.environ, "PORT": str(chosen_port)}
    # Resolves the start command AND validates we're inside a scaffolded project.
    web = procfile.web_argv(target, env=env)

    report: list[steps.Step] = [_install_step(target, no_install=no_install, use_uv=use_uv)]
    output.emit(report, lambda d: steps.render_steps(d, heading="Dev"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    command = _dev_command(target, web, use_uv=use_uv)
    url = f"http://localhost:{chosen_port}"
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
            ("Launch the app in the current directory", "aai dev"),
            ("Use a specific port", "aai dev --port 8000"),
            ("Launch without opening a browser", "aai dev --no-open"),
            ("Skip the dependency install step", "aai dev --no-install"),
        ]
    ),
)
def dev(
    ctx: typer.Context,
    port: int = typer.Option(3000, "--port", help="Local server port."),
    no_open: bool = typer.Option(False, "--no-open", help="Launch, but don't open the browser."),
    no_install: bool = typer.Option(
        False, "--no-install", help="Skip dependency install; launch directly."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Launch the dev server for the app in the current directory.

    Run this from inside a project created by `aai init`. It installs dependencies
    if needed, then starts the FastAPI server with live reload and opens the browser.
    """

    def body(_state: AppState, json_mode: bool) -> None:
        run_dev(port=port, no_install=no_install, no_open=no_open, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
