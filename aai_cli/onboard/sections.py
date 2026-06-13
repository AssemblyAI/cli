from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import assemblyai as aai
import typer

from aai_cli import config, environments, init_exec, output, transcribe_exec, transcribe_render
from aai_cli.commands import doctor as doctor_cmd
from aai_cli.commands import setup as setup_cmd
from aai_cli.context import AppState, persist_browser_login
from aai_cli.errors import CLIError
from aai_cli.init import runner
from aai_cli.onboard.prompter import Prompter


class SectionResult(Enum):
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class WizardContext:
    state: AppState
    profile: str
    json_mode: bool
    # Set by build_path when a template is scaffolded; launch_app (the wizard's last
    # section) reads it to start the dev server once the other sections are done.
    scaffolded: Path | None = None


def _has_key(profile: str) -> bool:
    return config.resolve_api_key_optional(profile=profile) is not None


def welcome(prompter: Prompter, _ctx: WizardContext) -> SectionResult:
    prompter.section("Welcome to AssemblyAI")
    prompter.note("This wizard signs you in, runs your first transcription, and helps you build.")
    return SectionResult.DONE


def auth(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    if _has_key(ctx.profile):
        prompter.note("Already signed in.")
        return SectionResult.SKIPPED
    prompter.section("Sign in")
    if not prompter.interactive:
        # A browser sign-in can't complete in a non-interactive/agent session: it
        # would bind a loopback port and block for two minutes on a callback no one
        # can produce. Stop here with the actionable next step instead of hanging.
        prompter.note(
            "No API key found, and this is a non-interactive session — "
            "browser sign-in can't complete here. Run `assembly login` in a terminal, "
            "or set ASSEMBLYAI_API_KEY."
        )
        return SectionResult.FAILED
    # Browser sign-in only: we deliberately don't offer an API-key paste here so a
    # secret never lands in the terminal scrollback or shell history.
    prompter.note(f"No account yet? Create one at {environments.active().signup_url}")
    persist_browser_login(ctx.profile, environments.active().name)
    return SectionResult.DONE


def first_request(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Your first transcription")
    api_key = config.resolve_api_key(profile=ctx.profile)
    source = prompter.text(
        "Audio file path or YouTube/podcast URL (or press Enter to transcribe a sample clip)",
        default="",
    ).strip()
    label = source or "the sample clip"
    try:
        with output.status(
            f"Transcribing {label}…", json_mode=ctx.json_mode, quiet=ctx.state.quiet
        ):
            transcript = transcribe_exec.run_transcription(
                api_key,
                source or None,
                sample=not source,
                transcription_config=aai.TranscriptionConfig(),
            )
    except CLIError as exc:
        output.error_console.print(output.fail(f"Transcription failed: {exc.message}"))
        return SectionResult.FAILED
    if not ctx.json_mode:  # --json owns stdout (the final summary); skip the human render
        transcribe_render.render_transcript_result(transcript, output.console)
    return SectionResult.DONE


_BUILD_CHOICES = [
    ("audio-transcription", "Audio transcription web app"),
    ("live-captions", "Live captions web app"),
    ("voice-agent", "Voice agent web app"),
    ("skip", "Skip — just the CLI for now"),
]


def _environment_summary(checks: list[doctor_cmd.Check]) -> str:
    """The closing line, computed from the actual statuses: doctor.render's
    all-or-nothing `ok` flag can't say "warnings only", which previously put
    "Everything looks good." right under a warning."""
    failed = sum(1 for c in checks if c["status"] == "fail")
    warned = sum(1 for c in checks if c["status"] == "warn")
    if failed:
        noun = "problem" if failed == 1 else "problems"
        return output.fail(f"{failed} {noun} found — see fixes above.")
    if warned:
        noun = "warning" if warned == 1 else "warnings"
        return output.warn(f"Ready — {warned} {noun} (only affects streaming/agent).")
    return output.success("Everything looks good.")


def _render_environment(checks: list[doctor_cmd.Check]) -> str:
    """The wizard's render of the doctor checks: doctor's own per-check lines, with
    the summary derived from what the checks actually reported."""
    lines = [output.heading("Environment check"), *doctor_cmd.render_check_lines(checks)]
    lines.append("  " + _environment_summary(checks))
    return "\n".join(lines)


def environment(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    checks = [
        doctor_cmd.check_python(),
        doctor_cmd.check_ffmpeg(),
        doctor_cmd.check_audio(),
    ]
    if not ctx.json_mode:  # --json owns stdout (the final summary); skip the human render
        # `_render_environment` prints its own "Environment check" heading, so we don't
        # call prompter.section here (that would show the title twice); just space it
        # from the previous section with a blank line.
        output.console.print()
        output.console.print(_render_environment(checks))
        prompter.note("Warnings here only affect live streaming and the voice agent.")
    if any(c["status"] == "fail" for c in checks):
        return SectionResult.FAILED
    return SectionResult.DONE


def build_path(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("What do you want to build?")
    choice = prompter.select("Pick a starting point (or skip)", _BUILD_CHOICES, default="skip")
    if choice == "skip":
        return SectionResult.SKIPPED
    if not prompter.confirm(f"Scaffold the '{choice}' app now?", default=True):
        prompter.note(f"You can run `assembly init {choice}` whenever you're ready.")
        return SectionResult.SKIPPED
    # launch=False: the dev server blocks until Ctrl-C, so launching here would stop
    # the remaining sections from running. launch_app starts it once the wizard is done.
    try:
        ctx.scaffolded = init_exec.run_init(
            init_exec.InitOptions(
                template=choice,
                directory=None,
                no_install=False,
                no_open=True,
                force=False,
                here=False,
                port=3000,
            ),
            ctx.state,
            json_mode=ctx.json_mode,
            launch=False,
        )
    except (CLIError, typer.Exit):
        output.error_console.print(output.fail(f"Could not scaffold '{choice}'."))
        return SectionResult.FAILED
    return SectionResult.DONE


def claude_code(prompter: Prompter, _ctx: WizardContext) -> SectionResult:
    prompter.section("Coding agent (optional)")
    if not prompter.confirm("Wire up Claude Code (docs MCP + skills)?", default=False):
        return SectionResult.SKIPPED
    steps = [
        setup_cmd.install_mcp("user", force=False),
        setup_cmd.install_skill(force=False),
        setup_cmd.install_cli_skill(force=False),
    ]
    output.console.print(setup_cmd.render({"steps": steps}))
    if any(s["status"] == "failed" for s in steps):
        return SectionResult.FAILED
    return SectionResult.DONE


def next_steps(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("You're set up")
    if not ctx.json_mode:  # --json owns stdout (the final summary); hints are human-only
        output.console.print(output.hint("Transcribe a file:  assembly transcribe <file>"))
        output.console.print(output.hint("Stream live audio:  assembly stream"))
        output.console.print(output.hint("Build an app:       assembly init"))
    return SectionResult.DONE


def launch_app(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    """Start the dev server for the app build_path scaffolded, like `assembly init` does.

    Must run as the wizard's final section: the server blocks until Ctrl-C, so any
    section after it would never run.
    """
    if ctx.scaffolded is None:
        return SectionResult.SKIPPED
    run_hint = f"cd {ctx.scaffolded} && assembly dev"
    if not prompter.interactive:
        prompter.note(f"Launch your app with `{run_hint}`.")
        return SectionResult.SKIPPED
    prompter.section("Launch your app")
    if not prompter.confirm("Start the dev server and open the browser now?", default=True):
        prompter.note(f"Launch it any time with `{run_hint}`.")
        return SectionResult.SKIPPED
    try:
        init_exec.launch_app(
            ctx.scaffolded,
            port=3000,
            use_uv=runner.has_uv(),
            no_open=False,
            json_mode=ctx.json_mode,
        )
    except (CLIError, typer.Exit):
        output.error_console.print(output.fail(f"The dev server didn't start. Try `{run_hint}`."))
        return SectionResult.FAILED
    return SectionResult.DONE
