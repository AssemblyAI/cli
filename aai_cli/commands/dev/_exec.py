"""Run logic for `assembly dev`: boot the scaffolded project's dev server.

The command module (aai_cli/commands/dev/__init__.py) only parses argv — it builds a
``DevOptions`` and hands it to ``run_dev`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the server orchestration by
constructing options directly instead of round-tripping argv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import output, steps
from aai_cli.context import AppState
from aai_cli.init import devserver, procfile, runner


@dataclass(frozen=True)
class DevOptions:
    """Every `assembly dev` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    port: int
    host: str
    no_install: bool
    no_open: bool


def run_dev(opts: DevOptions, state: AppState, *, json_mode: bool) -> None:
    """Boot the project's Procfile `web:` process locally, with live reload."""
    target = Path.cwd()
    use_uv = runner.has_uv()

    chosen_port = runner.find_free_port(opts.port)
    devserver.notify_port_change(opts.port, chosen_port, json_mode=json_mode, quiet=state.quiet)
    env = {**os.environ, "PORT": str(chosen_port)}
    # Resolves the start command AND validates we're inside a scaffolded project.
    web = procfile.web_argv(target, env=env)

    report: list[steps.Step] = [
        devserver.install_step(target, no_install=opts.no_install, use_uv=use_uv)
    ]
    output.emit(report, lambda d: steps.render_steps(d, heading="Dev"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    command = devserver.dev_command(target, web, use_uv=use_uv, host=opts.host)
    # The printed URL reflects the actual bind: "localhost" for the loopback
    # default, the literal host for an explicit --host.
    url_host = "localhost" if opts.host == devserver.LOCAL_HOST else opts.host
    url = f"http://{url_host}:{chosen_port}"
    if not json_mode:
        output.console.print(
            f"[aai.heading]Starting[/aai.heading] [aai.url]{escape(url)}[/aai.url]"
            "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
        )
    code = runner.run_server(
        target, command=command, port=chosen_port, env=env, open_browser=not opts.no_open
    )
    if code:
        raise typer.Exit(code=code)
