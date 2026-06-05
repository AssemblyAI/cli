# aai_cli/commands/init.py
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import environments, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.init import keys, runner, scaffold, templates

# Single-command sub-typer flattened to `aai init` (the exact pattern `aai transcribe`
# uses): one @app.command() named `init`, registered via app.add_typer(init.app) with
# no name. Bare `aai init` runs the command with template=None -> the interactive picker.
app = typer.Typer()


def _pick_template() -> str:
    """Interactive picker; raises a usage error when there's no TTY to prompt on."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise CLIError(
            "No template given and not running interactively. "
            f"Pass one of: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="usage_error",
            exit_code=1,
        )
    try:
        import questionary
    except ImportError as exc:  # a broken/stale install missing the declared dep
        raise CLIError(
            "The interactive picker needs 'questionary'. Reinstall the CLI "
            "(e.g. `uv tool install --reinstall aai-cli`), or pass a template "
            f"directly: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="missing_dependency",
            exit_code=1,
        ) from exc

    choice = questionary.select(
        "Pick a template",
        choices=[
            questionary.Choice(title=templates.title_for(t), value=t)
            for t in templates.TEMPLATE_ORDER
        ],
    ).ask()
    if choice is None:  # user pressed Ctrl-C
        raise typer.Exit(code=130)
    return str(choice)


def _resolve_dir(directory: str | None, template: str, *, here: bool) -> Path:
    if here:
        return Path.cwd()
    if directory:
        return Path(directory)
    return Path.cwd() / f"{template}-app"


@app.command()
def init(
    ctx: typer.Context,
    template: str = typer.Argument(None, help="Template to scaffold (omit to pick interactively)."),
    directory: str = typer.Argument(None, help="Target directory (default: <template>-app)."),
    no_install: bool = typer.Option(
        False, "--no-install", help="Scaffold only; don't install or launch."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Install + launch, but don't open the browser."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite a non-empty target directory."),
    here: bool = typer.Option(False, "--here", help="Scaffold into the current directory."),
    port: int = typer.Option(3000, "--port", help="Local server port."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Pick a template, scaffold it, install deps, launch the server, open the browser."""

    def body(state: AppState, json_mode: bool) -> None:
        chosen = template
        if chosen is None:
            chosen = _pick_template()
        if not templates.is_template(chosen):
            raise CLIError(
                f"Unknown template {chosen!r}. Choose one of: "
                f"{', '.join(templates.TEMPLATE_ORDER)}.",
                error_type="usage_error",
                exit_code=1,
            )

        if here and directory:
            raise CLIError(
                "Pass either a DIRECTORY or --here, not both.",
                error_type="usage_error",
                exit_code=1,
            )
        target = _resolve_dir(directory, chosen, here=here)
        if scaffold.target_conflict(target) and not force:
            raise CLIError(
                f"{target} already exists and is not empty. "
                f"Use --force to overwrite or pick another directory.",
                error_type="usage_error",
                exit_code=1,
            )

        api_key = keys.resolve_optional_api_key(profile=state.profile)
        # Pin the app to the active environment's hosts so a sandbox key (minted by
        # `aai login` against a non-prod env) isn't rejected by the production defaults.
        env = environments.active()
        env_vars = {
            "ASSEMBLYAI_BASE_URL": env.api_base,
            "ASSEMBLYAI_LLM_GATEWAY_URL": env.llm_gateway_base,
            "ASSEMBLYAI_STREAMING_HOST": env.streaming_host,
            # Voice Agent host mirrors the streaming host's naming across environments.
            "ASSEMBLYAI_AGENTS_HOST": env.streaming_host.replace("streaming", "agents", 1),
        }
        scaffold.scaffold(chosen, target, api_key=api_key, env_vars=env_vars)

        report: list[steps.Step] = [
            {"name": "scaffold", "status": "created", "detail": str(target)}
        ]
        if api_key is None:
            report.append(
                {
                    "name": "key",
                    "status": "skipped",
                    "detail": "no API key found; wrote a placeholder to .env (run `aai login`)",
                }
            )

        use_uv = runner.has_uv()
        will_launch = not no_install and api_key is not None
        if no_install:
            report.append({"name": "install", "status": "skipped", "detail": "--no-install"})
        else:
            setup = runner.run_setup(target, use_uv=use_uv)
            if setup.returncode != 0:
                report.append(
                    {
                        "name": "install",
                        "status": "failed",
                        "detail": (setup.stderr or setup.stdout).strip()[:300],
                    }
                )
                will_launch = False
            else:
                report.append(
                    {
                        "name": "install",
                        "status": "installed",
                        "detail": "uv" if use_uv else "venv + pip",
                    }
                )

        # Deps are installed but there's no key, so the server can't start — say so
        # rather than exiting silently.
        if not no_install and api_key is None:
            report.append(
                {
                    "name": "launch",
                    "status": "skipped",
                    "detail": f"no API key; run `aai login`, then: cd {target} && uv run uvicorn api.index:app",
                }
            )

        output.emit(
            report, lambda d: steps.render_steps(d, heading="aai init:"), json_mode=json_mode
        )
        if any(s["status"] == "failed" for s in report):
            raise typer.Exit(code=1)

        if will_launch:
            chosen_port = runner.find_free_port(port)
            url = f"http://localhost:{chosen_port}"
            if not json_mode:
                output.console.print(
                    f"[aai.heading]Starting[/aai.heading] {escape(url)}  (Ctrl-C to stop)"
                )
            code = runner.launch_and_open(
                target, port=chosen_port, use_uv=use_uv, open_browser=not no_open
            )
            if code:
                raise typer.Exit(code=code)

    run_command(ctx, body, json=json_out)
