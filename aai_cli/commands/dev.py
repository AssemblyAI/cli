# aai_cli/commands/dev.py
from __future__ import annotations

from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import help_panels, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog
from aai_cli.init import runner

# Flattened single-command sub-typer (same pattern as `aai init`): one
# @app.command() registered via app.add_typer(dev.app) with no name.
app = typer.Typer()


def _require_app_dir() -> Path:
    """The current directory, if it holds a scaffolded FastAPI app (`api/index.py`)."""
    cwd = Path.cwd()
    if not (cwd / "api" / "index.py").exists():
        raise CLIError(
            "No app found here (expected api/index.py). cd into a project created by "
            "`aai init`, or run `aai init` to scaffold one.",
            error_type="usage_error",
            exit_code=1,
        )
    return cwd


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


def run_dev(*, port: int, no_install: bool, no_open: bool, json_mode: bool) -> None:
    """Install deps if needed, then launch the dev server with live reload."""
    target = _require_app_dir()
    use_uv = runner.has_uv()

    report: list[steps.Step] = [_install_step(target, no_install=no_install, use_uv=use_uv)]
    output.emit(report, lambda d: steps.render_steps(d, heading="Dev"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    chosen_port = runner.find_free_port(port)
    url = f"http://localhost:{chosen_port}"
    if not json_mode:
        output.console.print(
            f"[aai.heading]Starting[/aai.heading] [aai.url]{escape(url)}[/aai.url]"
            "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
        )
    code = runner.launch_and_open(
        target, port=chosen_port, use_uv=use_uv, open_browser=not no_open, reload=True
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
