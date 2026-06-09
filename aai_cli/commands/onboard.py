from __future__ import annotations

import sys

import typer

from aai_cli import config, help_panels, output
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.help_text import examples_epilog
from aai_cli.onboard import progress, wizard
from aai_cli.onboard.prompter import InteractivePrompter, NonInteractivePrompter, Prompter
from aai_cli.onboard.sections import WizardContext

app = typer.Typer()


def build_prompter() -> Prompter:
    """A real prompter only when both ends are a TTY; otherwise never block."""
    if sys.stdin.isatty() and sys.stdout.isatty():
        return InteractivePrompter()
    return NonInteractivePrompter()


@app.command(
    rich_help_panel=help_panels.QUICK_START,
    epilog=examples_epilog(
        [
            ("Run the guided setup", "aai onboard"),
            ("Show your progress toward 100 requests", "aai onboard --status"),
        ]
    ),
)
def onboard(
    ctx: typer.Context,
    status: bool = typer.Option(False, "--status", help="Show request progress and exit."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Guided setup: sign in, run your first transcription, and start building."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        if status:
            count = config.get_requests_made(profile)
            output.emit(
                {"requests_made": count, "goal": progress.GOAL},
                lambda _d: progress.render_progress(count),
                json_mode=json_mode,
            )
            return
        wiz_ctx = WizardContext(state=state, profile=profile, json_mode=json_mode)
        code = wizard.run_onboarding(build_prompter(), wiz_ctx)
        if code != 0:
            raise typer.Exit(code=code)

    # auto_login=False: the wizard owns the sign-in step itself.
    run_command(ctx, body, json=json_out, auto_login=False)
