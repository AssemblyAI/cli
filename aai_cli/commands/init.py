# aai_cli/commands/init.py
from __future__ import annotations

import typer

from aai_cli import help_panels, init_exec, options
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.init import templates

# Single-command sub-typer flattened to `assembly init` (the exact pattern `assembly transcribe`
# uses): one @app.command() named `init`, registered via app.add_typer(init.app) with
# no name. Bare `assembly init` runs the command with template=None -> the interactive picker.
app = typer.Typer()


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
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Overwrite a non-empty target directory (overlays the template; "
            "files not in the template are kept)."
        ),
    ),
    here: bool = typer.Option(False, "--here", help="Scaffold into the current directory."),
    port: int = typer.Option(init_exec.DEFAULT_PORT, "--port", help="Local server port."),
    json_out: bool = options.json_option(),
) -> None:
    """Scaffold a new project from a template, then launch it.

    This is the starting point for creating an app — including a voice agent app
    ('assembly init voice-agent'). The 'assembly agent' command only runs a live mic
    conversation and writes no code.
    """
    opts = init_exec.InitOptions(
        template=template,
        directory=directory,
        no_install=no_install,
        no_open=no_open,
        force=force,
        here=here,
        port=port,
    )

    # run_init returns the scaffolded path (for the onboarding wizard); the command
    # path discards it, so a thin body adapts the run_command (-> None) signature.
    def body(state: AppState, json_mode: bool) -> None:
        init_exec.run_init(opts, state, json_mode=json_mode)

    run_command(ctx, body, json=json_out)
