# aai_cli/commands/share.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import config, help_panels, options, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog
from aai_cli.init import devserver, procfile, runner, tunnel

# Flattened single-command sub-typer (same pattern as `aai dev`).
app = typer.Typer()


def _require_cloudflared() -> None:
    if shutil.which(tunnel.CLOUDFLARED) is None:
        raise CLIError(
            "cloudflared is required to share a public link.",
            error_type="missing_dependency",
            exit_code=1,
            suggestion="Install it: brew install cloudflared",
        )


def _render_share(data: dict[str, object]) -> str:
    return (
        f"[aai.heading]Sharing[/aai.heading] [aai.url]{escape(str(data['url']))}[/aai.url]\n"
        f"[aai.muted]→ serving[/aai.muted] [aai.url]{escape(str(data['local']))}[/aai.url]"
        "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
    )


def _terminate(proc: subprocess.Popen[str] | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()


def run_share(*, port: int, no_install: bool, json_mode: bool) -> None:
    """Boot the app and expose it on a public cloudflared quick-tunnel URL."""
    target = Path.cwd()
    use_uv = runner.has_uv()

    chosen_port = runner.find_free_port(port)
    env = {**os.environ, "PORT": str(chosen_port)}
    web = procfile.web_argv(target, env=env)  # validates we're in a scaffolded project
    _require_cloudflared()

    report: list[steps.Step] = [
        devserver.install_step(target, no_install=no_install, use_uv=use_uv)
    ]
    output.emit(report, lambda d: steps.render_steps(d, heading="Share"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    server = runner.spawn(devserver.dev_command(target, web, use_uv=use_uv), cwd=target, env=env)
    proxy: subprocess.Popen[str] | None = None
    try:
        if not runner.wait_for_port(chosen_port):
            raise CLIError(
                "The dev server didn't start, so there's nothing to share.",
                error_type="server_error",
                exit_code=1,
            )
        fd, name = tempfile.mkstemp(prefix="aai-tunnel-", suffix=".log")
        os.close(fd)
        log_path = Path(name)
        # The tunnel binary only proxies the port; don't hand it the API key the
        # dev server needs (keeps the secret out of cloudflared's logs/diagnostics).
        tunnel_env = {k: v for k, v in os.environ.items() if k != config.ENV_API_KEY}
        proxy = runner.spawn(
            tunnel.tunnel_command(chosen_port), cwd=target, env=tunnel_env, log_path=log_path
        )
        public = tunnel.await_url(log_path)
        if public is None:
            raise CLIError(
                "cloudflared didn't report a tunnel URL in time.",
                error_type="tunnel_error",
                exit_code=1,
            )
        payload: dict[str, object] = {
            "url": public,
            "local": f"http://localhost:{chosen_port}",
            "port": chosen_port,
        }
        output.emit(payload, _render_share, json_mode=json_mode)
        server.wait()
    except KeyboardInterrupt:
        pass
    finally:
        _terminate(proxy)
        _terminate(server)


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Share the running app on a public URL", "aai share"),
            ("Use a specific local port", "aai share --port 8000"),
            ("Skip the dependency install step", "aai share --no-install"),
        ]
    ),
)
def share(
    ctx: typer.Context,
    port: int = typer.Option(3000, "--port", help="Local server port."),
    no_install: bool = typer.Option(
        False, "--no-install", help="Skip dependency install; launch directly."
    ),
    json_out: bool = options.json_option(),
) -> None:
    """Boot the app and expose it on a public URL via a cloudflared tunnel.

    Run this from inside a project created by `aai init`. It starts the dev server and
    opens a cloudflared quick tunnel, printing a shareable https://*.trycloudflare.com
    URL. Requires cloudflared (`brew install cloudflared`).
    """

    def body(_state: AppState, json_mode: bool) -> None:
        run_share(port=port, no_install=no_install, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
