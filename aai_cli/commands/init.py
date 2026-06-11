# aai_cli/commands/init.py
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.markup import escape

from aai_cli import __version__, environments, help_panels, options, output, steps
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.help_text import examples_epilog
from aai_cli.init import keys, runner, scaffold, templates

# Single-command sub-typer flattened to `assembly init` (the exact pattern `assembly transcribe`
# uses): one @app.command() named `init`, registered via app.add_typer(init.app) with
# no name. Bare `assembly init` runs the command with template=None -> the interactive picker.
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
    return Path.cwd() / template


def _resolve_template(template: str | None) -> str:
    """Resolve the template name: the picker when omitted, else validate the arg."""
    chosen = template if template is not None else _pick_template()
    if not templates.is_template(chosen):
        raise CLIError(
            f"Unknown template {chosen!r}. Choose one of: {', '.join(templates.TEMPLATE_ORDER)}.",
            error_type="usage_error",
            exit_code=1,
        )
    return chosen


def _active_env_vars() -> dict[str, str]:
    """Pin the scaffolded app to the active environment's hosts.

    A sandbox key (minted by `assembly login` against a non-prod env) would otherwise be
    rejected by the production defaults baked into the template.
    """
    env = environments.active()
    return {
        "ASSEMBLYAI_BASE_URL": env.api_base,
        "ASSEMBLYAI_LLM_GATEWAY_URL": env.llm_gateway_base,
        "ASSEMBLYAI_STREAMING_HOST": env.streaming_host,
        # Voice Agent host mirrors the streaming host's naming across environments.
        "ASSEMBLYAI_AGENTS_HOST": env.streaming_host.replace("streaming", "agents", 1),
    }


def _install_step(
    target: Path, *, no_install: bool, api_key: str | None, use_uv: bool
) -> tuple[list[steps.Step], bool]:
    """Run (or skip) dependency install, returning the report rows and whether to launch.

    Launch only happens when deps are installed and there's a key; an install failure
    flips `will_launch` off so the caller exits non-zero instead of starting a server.
    """
    will_launch = not no_install and api_key is not None
    if no_install:
        return [{"name": "install", "status": "skipped", "detail": "--no-install"}], will_launch
    setup = runner.run_setup(target, use_uv=use_uv)
    if setup.returncode != 0:
        row: steps.Step = {
            "name": "install",
            "status": "failed",
            "detail": (setup.stderr or setup.stdout).strip()[:300],
        }
        # The False (don't-launch) is an equivalent mutant: run_init raises Exit(1) on
        # any failed step before it ever consults will_launch, so the value is unused
        # on this branch.
        return [row], False  # pragma: no mutate
    return [
        {
            "name": "install",
            "status": "installed",
            "detail": "uv" if use_uv else "venv + pip",
        }
    ], will_launch


def _resolve_target(directory: str | None, chosen: str, *, here: bool, force: bool) -> Path:
    """Resolve the target directory and reject --here+DIRECTORY or a non-empty conflict."""
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
    return target


def _scaffold_report(chosen: str, target: Path, api_key: str | None) -> list[steps.Step]:
    """Write the template to `target` and return the opening report rows."""
    scaffold.scaffold(chosen, target, api_key=api_key, env_vars=_active_env_vars())
    report: list[steps.Step] = [{"name": "scaffold", "status": "created", "detail": str(target)}]
    if api_key is None:
        report.append(
            {
                "name": "key",
                "status": "skipped",
                "detail": "no API key found; wrote a placeholder to .env (run `assembly login`)",
            }
        )
    return report


def launch_app(target: Path, *, port: int, use_uv: bool, no_open: bool, json_mode: bool) -> None:
    """Start the scaffolded app on a free port and open the browser, then block.

    Public (not underscore-private) because the onboarding wizard launches the
    scaffolded app as its final step, after the remaining wizard sections have run.
    """
    chosen_port = runner.find_free_port(port)
    url = f"http://localhost:{chosen_port}"
    if not json_mode:
        output.console.print(
            f"[aai.heading]Starting[/aai.heading] [aai.url]{escape(url)}[/aai.url]"
            "  [aai.muted](Ctrl-C to stop)[/aai.muted]"
        )
    code = runner.launch_and_open(target, port=chosen_port, use_uv=use_uv, open_browser=not no_open)
    if code:
        raise typer.Exit(code=code)


def run_init(
    state: AppState,
    *,
    template: str | None,
    directory: str | None,
    no_install: bool,
    no_open: bool,
    force: bool,
    here: bool,
    port: int,
    json_mode: bool,
    launch: bool = True,
) -> Path:
    """Scaffold (and optionally install/launch) a template; return the target dir.

    `launch=False` is for callers like the onboarding wizard that must not block on a
    running dev server mid-flow — it stops after install and leaves the run command as
    a hint (the wizard calls `launch_app` itself once its remaining sections are done).
    """
    if not json_mode:
        # Vercel-style banner at the top of the run. Decoration goes to stderr (data →
        # stdout): it must never pollute a piped stdout, even on an error path.
        output.error_console.print(
            f"[aai.heading]AssemblyAI CLI[/aai.heading] [aai.muted]{__version__}[/aai.muted]"
        )
    chosen = _resolve_template(template)
    target = _resolve_target(directory, chosen, here=here, force=force)

    api_key = keys.resolve_optional_api_key(profile=state.profile)
    report = _scaffold_report(chosen, target, api_key)

    use_uv = runner.has_uv()
    install_rows, will_launch = _install_step(
        target, no_install=no_install, api_key=api_key, use_uv=use_uv
    )
    report.extend(install_rows)

    # Deps are installed but there's no key, so the server can't start — say so
    # rather than exiting silently.
    if not no_install and api_key is None:
        report.append(
            {
                "name": "launch",
                "status": "skipped",
                "detail": f"no API key; run `assembly login`, then: cd {target} && assembly dev",
            }
        )

    output.emit(report, lambda d: steps.render_steps(d, heading="Setup"), json_mode=json_mode)
    if any(s["status"] == "failed" for s in report):
        raise typer.Exit(code=1)

    if launch and will_launch:
        launch_app(target, port=port, use_uv=use_uv, no_open=no_open, json_mode=json_mode)
    elif not json_mode:
        # Scaffolded but not launched (no key, or --no-install, or launch=False): leave the
        # user with the one command that starts their app, the way `vercel`/`supabase` sign off.
        output.console.print(output.hint(f"Run `cd {escape(str(target))} && assembly dev`."))
    return target


@app.command(
    rich_help_panel=help_panels.BUILD,
    epilog=examples_epilog(
        [
            ("Scaffold a new app interactively", "assembly init"),
            (
                "Scaffold an audio transcription app into ./my-app",
                "assembly init audio-transcription my-app",
            ),
            ("Scaffold a voice agent app", "assembly init voice-agent"),
            ("Scaffold into the current directory", "assembly init audio-transcription --here"),
            (
                "Scaffold only, without installing or launching",
                "assembly init audio-transcription --no-install",
            ),
        ]
    ),
)
def init(
    ctx: typer.Context,
    template: str | None = typer.Argument(
        None,
        # Enumerate the registry so the help text can never drift from the templates
        # that actually ship.
        help=(
            f"Template to scaffold: {', '.join(templates.TEMPLATE_ORDER)} "
            "(omit to pick interactively)."
        ),
    ),
    directory: str | None = typer.Argument(None, help="Target directory (default: <template>)."),
    no_install: bool = typer.Option(
        False, "--no-install", help="Scaffold only; don't install or launch."
    ),
    no_open: bool = typer.Option(
        False, "--no-open", help="Install + launch, but don't open the browser."
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite a non-empty target directory."),
    here: bool = typer.Option(False, "--here", help="Scaffold into the current directory."),
    port: int = typer.Option(3000, "--port", help="Local server port."),
    json_out: bool = options.json_option(),
) -> None:
    """Scaffold a new project from a template, then launch it.

    This is the starting point for creating an app — including a voice agent app
    ('assembly init voice-agent'). The 'assembly agent' command only runs a live mic
    conversation and writes no code.
    """

    def body(state: AppState, json_mode: bool) -> None:
        run_init(
            state,
            template=template,
            directory=directory,
            no_install=no_install,
            no_open=no_open,
            force=force,
            here=here,
            port=port,
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
