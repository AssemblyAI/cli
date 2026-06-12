from __future__ import annotations

from collections.abc import Callable

from aai_cli import output
from aai_cli.errors import NotAuthenticated
from aai_cli.onboard import sections
from aai_cli.onboard.prompter import Prompter, WizardCancelled
from aai_cli.onboard.sections import SectionResult, WizardContext

_SectionFn = Callable[[Prompter, WizardContext], SectionResult]


def run_onboarding(prompter: Prompter, ctx: WizardContext) -> int:
    """Run the ordered sections; return a process exit code.

    Auth is the one hard stop (no key → later sections can't run); any other failed
    section is recorded and surfaced in the closing line, with exit code 1, instead
    of being declared a success. Cancellation (Ctrl-C / empty pick) exits cleanly.
    The terminal cursor is always restored.
    """
    results: dict[str, str] = {}

    def _run(label: str, section: _SectionFn) -> SectionResult:
        result = section(prompter, ctx)
        results[label] = result.value
        return result

    try:
        _run("welcome", sections.welcome)
        if _run("sign-in", sections.auth) is SectionResult.FAILED:
            # The auth section already printed the specific next step (browser retry,
            # or — non-interactively — `assembly login`/ASSEMBLYAI_API_KEY), so keep this
            # terminal line neutral rather than implying a re-run always fixes it.
            if not ctx.json_mode:
                output.error_console.print(output.fail("Sign-in didn't complete."))
            return _summarize(ctx, results, NotAuthenticated().exit_code)
        _run("first transcription", sections.first_request)
        _run("environment", sections.environment)
        _run("build path", sections.build_path)
        _run("coding agent", sections.claude_code)
        _run("next steps", sections.next_steps)
        # Last on purpose: the dev server blocks until Ctrl-C.
        _run("launch app", sections.launch_app)
    except WizardCancelled:
        output.error_console.print(
            output.hint("Setup cancelled. Run `assembly onboard` to resume.")
        )
        return 130
    else:
        return _summarize(ctx, results, 0)
    finally:
        output.console.show_cursor(show=True)


def _final_code(results: dict[str, str], code: int) -> tuple[list[str], int]:
    """The failed section names, and the exit code they imply.

    Any failed section turns a would-be-0 exit into 1; a harder failure (the auth
    stop's 4) keeps its own code.
    """
    failed = [name for name, value in results.items() if value == SectionResult.FAILED.value]
    if failed and code == 0:
        code = 1
    return failed, code


def _failure_line(failed: list[str]) -> str:
    noun = "issue" if len(failed) == 1 else "issues"
    return f"Set up with {len(failed)} {noun} ({', '.join(failed)} failed)."


def _summarize(ctx: WizardContext, results: dict[str, str], code: int) -> int:
    """Fold the per-section results into the closing output and final exit code.

    Under --json the summary is the one stdout payload; human runs get a closing
    stderr line naming what failed.
    """
    failed, code = _final_code(results, code)
    if ctx.json_mode:
        output.emit_ndjson(
            {"ok": code == 0, "exit_code": code, "sections": results, "failed": failed}
        )
    elif failed:
        output.error_console.print(output.fail(_failure_line(failed)))
    return code
