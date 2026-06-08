from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import assemblyai as aai

from aai_cli import client, config, environments, output, transcribe_render
from aai_cli.context import AppState, persist_browser_login
from aai_cli.errors import NotAuthenticated
from aai_cli.onboard import progress
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


def welcome(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    count = config.get_requests_made(ctx.profile)
    if count:
        prompter.section("Welcome back to AssemblyAI")
        output.console.print(progress.render_progress(count))
        return SectionResult.DONE
    prompter.section("Welcome to AssemblyAI")
    prompter.note("This wizard signs you in, runs your first transcription, and helps you build.")
    return SectionResult.DONE


def auth(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    if _has_key(ctx.profile):
        prompter.note("Already signed in.")
        return SectionResult.SKIPPED
    prompter.section("Sign in")
    method = prompter.select(
        "How do you want to sign in?",
        [("browser", "Sign in with your browser (recommended)"), ("key", "Paste an API key")],
        default="browser",
    )
    env = environments.active().name
    if method == "key":
        key = prompter.text("Paste your AssemblyAI API key")
        if not client.validate_key(key):
            output.error_console.print(output.fail("That key was rejected."))
            return SectionResult.FAILED
        config.set_api_key(ctx.profile, key)
        config.set_profile_env(ctx.profile, env)
        return SectionResult.DONE
    prompter.note(f"No account yet? Create one at {environments.active().signup_url}")
    persist_browser_login(ctx.profile, env)
    return SectionResult.DONE


def first_request(prompter: Prompter, ctx: WizardContext) -> SectionResult:
    prompter.section("Your first transcription")
    api_key = config.resolve_api_key(profile=ctx.profile)
    with output.status("Transcribing the sample clip…", json_mode=ctx.json_mode):
        transcript = client.transcribe(
            api_key, client.SAMPLE_AUDIO_URL, config=aai.TranscriptionConfig()
        )
    count = config.record_request(ctx.profile)
    transcribe_render.render_transcript_result(transcript, output.console)
    output.console.print(progress.render_progress(count))
    return SectionResult.DONE
