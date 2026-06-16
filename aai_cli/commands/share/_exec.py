"""Run logic for `assembly share`: expose the local dev server on a public URL.

The command module (aai_cli/commands/share/__init__.py) only parses argv — it builds a
``ShareOptions`` and hands it to ``run_share`` via ``context.run_command`` (the
options/run split, see AGENTS.md), so tests drive the tunnel orchestration by
constructing options directly instead of round-tripping argv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli.app.context import AppState
from aai_cli.core import env as os_env
from aai_cli.core import errors
from aai_cli.core.errors import CLIError
from aai_cli.init import devserver, procfile, runner, tunnel
from aai_cli.ui import output, steps


@dataclass(frozen=True)
class ShareOptions:
    """Every `assembly share` flag as plain data (``--json`` excluded: run_command
    resolves it into the ``json_mode`` argument)."""

    port: int
    no_install: bool


def _render_share(data: dict[str, object]) -> str:
    return (
        f"[aai.heading]Sharing[/aai.heading] [aai.url]{escape(str(data['url']))}[/aai.url]\n"
        f"[aai.muted]→ serving[/aai.muted] [aai.url]{escape(str(data['local']))}[/aai.url]"
        "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
    )


def run_share(opts: ShareOptions, state: AppState, *, json_mode: bool) -> None:
    """Boot the app and expose it on a public cloudflared quick-tunnel URL."""
    target = Path.cwd()
    use_uv = runner.has_uv()

    chosen_port = runner.find_free_port(opts.port)
    devserver.notify_port_change(opts.port, chosen_port, json_mode=json_mode, quiet=state.quiet)
    env = os_env.child_env(PORT=str(chosen_port))
    web = procfile.web_argv(target, env=env)  # validates we're in a scaffolded project
    tunnel.require_cloudflared("share a public link")

    report: list[steps.Step] = [
        devserver.install_step(target, no_install=opts.no_install, use_uv=use_uv)
    ]
    output.emit(report, lambda d: steps.render_steps(d, heading="Share"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    server = runner.spawn(devserver.dev_command(target, web, use_uv=use_uv), cwd=target, env=env)
    proxy = None
    log_path: Path | None = None
    keep_log = False
    try:
        if not runner.wait_for_port(chosen_port):
            raise CLIError(
                "The dev server didn't start, so there's nothing to share.",
                error_type="server_error",
                exit_code=1,
            )
        proxy, public, log_path = tunnel.open_quick_tunnel(chosen_port, cwd=target)
        if public is None:
            # Keep the captured cloudflared output: it's the only evidence of why
            # the tunnel never came up.
            keep_log = True
            raise CLIError(
                "cloudflared didn't report a tunnel URL in time.",
                error_type="tunnel_error",
                exit_code=1,
                suggestion=f"cloudflared's output was kept at {log_path} — check it for errors.",
            )
        payload: dict[str, object] = {
            "url": public,
            "local": f"http://localhost:{chosen_port}",
            "port": chosen_port,
        }
        output.emit(payload, _render_share, json_mode=json_mode)
        server.wait()
    except KeyboardInterrupt:
        # Ctrl-C is the expected way to stop a foreground share: tear down (finally,
        # below) then exit 130 (cancel) so it isn't reported to a caller as success.
        raise typer.Exit(code=errors.CANCELLED_EXIT_CODE) from None
    finally:
        tunnel.terminate(proxy)
        tunnel.terminate(server)
        if log_path is not None and not keep_log:
            log_path.unlink(missing_ok=True)
