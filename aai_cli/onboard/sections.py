from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import assemblyai as aai
import typer

from aai_cli import config, environments, output, transcribe_render
from aai_cli.commands import doctor as doctor_cmd
from aai_cli.commands import init as init_cmd
from aai_cli.commands import setup as setup_cmd
from aai_cli.commands import transcribe as transcribe_cmd
from aai_cli.context import AppState, persist_browser_login
from aai_cli.errors import CLIError, NotAuthenticated
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


def _has_key(profile: str) -> bool:
    try:
        config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return False
    return True


def welcome(prompter: Prompter, _ctx: WizardContext) -> SectionResult:
    prompter.section("Welcome to AssemblyAI")
    prompter.note("This wizard signs you in, runs your first transcription, and helps you build.")
    return SectionResult.DONE


def auth(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    if _has_key(ctx.profile):
        prompter.note("Already signed in.")
        return SectionResult.SKIPPED
    prompter.section("Sign in")
    # Browser sign-in only: we deliberately don't offer an API-key paste here so a
    # secret never lands in the terminal scrollback or shell history.
    prompter.note(f"No account yet? Create one at {environments.active().signup_url}")
    persist_browser_login(ctx.profile, environments.active().name)
    return SectionResult.DONE


def first_request(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Your first transcription")
    api_key = config.resolve_api_key(profile=ctx.profile)
    source = prompter.text(
        "Audio file path or YouTube URL (or press Enter to transcribe a sample clip)",
        default="",
    ).strip()
    label = source or "the sample clip"
    try:
        with output.status(f"Transcribing {label}…", json_mode=ctx.json_mode):
            transcript = transcribe_cmd._transcribe_audio(  # pyright: ignore[reportPrivateUsage]
                api_key,
                source or None,
                sample=not source,
                transcription_config=aai.TranscriptionConfig(),
            )
    except CLIError as exc:
        output.error_console.print(output.fail(f"Transcription failed: {exc.message}"))
        return SectionResult.FAILED
    transcribe_render.render_transcript_result(transcript, output.console)
    return SectionResult.DONE


_BUILD_CHOICES = [
    ("audio-transcription", "Audio transcription web app"),
    ("live-captions", "Live captions web app"),
    ("voice-agent", "Voice agent web app"),
    ("skip", "Skip — just the CLI for now"),
]


def environment(prompter: Prompter, _ctx: WizardContext) -> SectionResult:
    checks = [
        doctor_cmd._check_python(),  # pyright: ignore[reportPrivateUsage]
        doctor_cmd._check_ffmpeg(),  # pyright: ignore[reportPrivateUsage]
        doctor_cmd._check_audio(),  # pyright: ignore[reportPrivateUsage]
    ]
    # `_render` already prints its own "Environment check" heading, so we don't call
    # prompter.section here (that would show the title twice); just space it from the
    # previous section with a blank line.
    output.console.print()
    output.console.print(doctor_cmd._render({"ok": True, "checks": checks}))  # pyright: ignore[reportPrivateUsage]
    prompter.note("Warnings here only affect live streaming and the voice agent.")
    return SectionResult.DONE


def build_path(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("What do you want to build?")
    choice = prompter.select("Pick a starting point (or skip)", _BUILD_CHOICES, default="skip")
    if choice == "skip":
        return SectionResult.SKIPPED
    if not prompter.confirm(f"Scaffold the '{choice}' app now?", default=True):
        prompter.note(f"You can run `aai init {choice}` whenever you're ready.")
        return SectionResult.SKIPPED
    # launch=False: never block the wizard on a running dev server.
    try:
        init_cmd.run_init(
            ctx.state,
            template=choice,
            directory=None,
            no_install=False,
            no_open=True,
            force=False,
            here=False,
            port=3000,
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
        setup_cmd._install_mcp("user", force=False),  # pyright: ignore[reportPrivateUsage]
        setup_cmd._install_skill(force=False),  # pyright: ignore[reportPrivateUsage]
        setup_cmd._install_cli_skill(force=False),  # pyright: ignore[reportPrivateUsage]
    ]
    output.console.print(setup_cmd._render({"steps": steps}))  # pyright: ignore[reportPrivateUsage]
    if any(s["status"] == "failed" for s in steps):
        return SectionResult.FAILED
    return SectionResult.DONE


def next_steps(prompter: Prompter, _ctx: WizardContext) -> SectionResult:
    prompter.section("You're set up")
    output.console.print(output.hint("Transcribe a file:  aai transcribe <file>"))
    output.console.print(output.hint("Stream live audio:  aai stream"))
    output.console.print(output.hint("Build an app:       aai init"))
    return SectionResult.DONE
