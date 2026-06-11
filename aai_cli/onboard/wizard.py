from __future__ import annotations

from aai_cli import output
from aai_cli.errors import NotAuthenticated
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import Prompter, WizardCancelled
from aai_cli.onboard.sections import SectionResult, WizardContext


def run_onboarding(prompter: Prompter, ctx: WizardContext) -> int:
    """Run the ordered sections; return a process exit code.

    Auth is the one hard stop (no key → later sections can't run). Cancellation
    (Ctrl-C / empty pick) exits cleanly. The terminal cursor is always restored.
    """
    try:
        sections.welcome(prompter, ctx)
        if sections.auth(prompter, ctx) is SectionResult.FAILED:
            # The auth section already printed the specific next step (browser retry,
            # or — non-interactively — `assembly login`/ASSEMBLYAI_API_KEY), so keep this
            # terminal line neutral rather than implying a re-run always fixes it.
            output.error_console.print(output.fail("Sign-in didn't complete."))
            return NotAuthenticated().exit_code
        sections.first_request(prompter, ctx)
        sections.environment(prompter, ctx)
        sections.build_path(prompter, ctx)
        sections.claude_code(prompter, ctx)
        sections.next_steps(prompter, ctx)
    except WizardCancelled:
        output.error_console.print(
            output.hint("Setup cancelled. Run `assembly onboard` to resume.")
        )
        return 130
    else:
        return 0
    finally:
        output.console.show_cursor(show=True)
