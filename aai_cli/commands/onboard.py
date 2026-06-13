from __future__ import annotations

import typer

from aai_cli import command_registry, help_panels, options
from aai_cli.app.context import AppState, run_command
from aai_cli.core import stdio
from aai_cli.core.errors import CLIError
from aai_cli.onboard import wizard
from aai_cli.onboard.prompter import InteractivePrompter, NonInteractivePrompter, Prompter
from aai_cli.onboard.sections import WizardContext
from aai_cli.ui import output
from aai_cli.ui.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.QUICK_START,
    order=10,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("onboard",),
)


def build_prompter(*, non_interactive: bool = False) -> Prompter:
    """A real prompter only when the caller hasn't opted out and both ends are a TTY;
    otherwise never block for input."""
    if non_interactive:
        return NonInteractivePrompter()
    if stdio.interactive_stdio():
        return InteractivePrompter()
    return NonInteractivePrompter()


@app.command(
    rich_help_panel=help_panels.QUICK_START,
    epilog=examples_epilog(
        [
            ("Run the guided setup", "assembly onboard"),
        ]
    ),
)
def onboard(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without interactive prompts (default when agent detected)",
    ),
) -> None:
    """Guided setup: sign in and run your first transcription"""

    def body(state: AppState, json_mode: bool) -> None:
        profile = state.resolve_profile()
        wiz_ctx = WizardContext(state=state, profile=profile, json_mode=json_mode)
        # --json also forces non-interactive: a machine-output run can't block on
        # prompts, and the interactive prompter would write prose onto the JSON stdout.
        forced = non_interactive or output.is_agentic() or json_mode
        code = wizard.run_onboarding(build_prompter(non_interactive=forced), wiz_ctx)
        if code != 0:
            if json_mode:
                # The standard {"error": …} envelope on stderr; the wizard already
                # emitted its JSON section summary on stdout.
                raise CLIError(
                    "Onboarding did not complete.",
                    error_type="onboarding_incomplete",
                    exit_code=code,
                )
            raise typer.Exit(code=code)

    # auto_login=False: the wizard owns the sign-in step itself.
    run_command(ctx, body, json=json_out, auto_login=False)
