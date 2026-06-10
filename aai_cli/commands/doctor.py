from __future__ import annotations

import shutil
import sys
from collections.abc import Mapping, Sequence
from typing import Protocol, TypedDict

import typer
from rich.markup import escape

from aai_cli import client, config, help_panels, options, output, theme
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.errors import CLIError, NotAuthenticated
from aai_cli.help_text import examples_epilog

app = typer.Typer()


class Check(TypedDict):
    """One diagnostic: a named check, its status, what it affects, and how to fix it."""

    name: str
    status: str  # "ok" | "warn" | "fail" — only "fail" makes `doctor` exit non-zero
    affects: list[str]
    detail: str
    fix: str | None


class DoctorResult(TypedDict):
    ok: bool
    checks: list[Check]


class _SoundDeviceModule(Protocol):
    def query_devices(self) -> Sequence[Mapping[str, object]]: ...


# Status -> (affordance symbol, render style). "fail" is a blocker; "warn" is
# degraded-but-usable. Drives the per-check glyph in `render`.
_SYMBOL = {
    "ok": (theme.SYMBOL_SUCCESS, "aai.success"),
    "warn": (theme.SYMBOL_WARN, "aai.warn"),
    "fail": (theme.SYMBOL_ERROR, "aai.error"),
}


def _check(
    name: str,
    status: str,
    detail: str,
    *,
    fix: str | None = None,
    affects: list[str] | None = None,
) -> Check:
    """Assemble a Check. ``affects`` defaults to empty — an 'ok' check blocks nothing."""
    return {"name": name, "status": status, "affects": affects or [], "detail": detail, "fix": fix}


def check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 12):
        return _check("python", "ok", version)
    return _check(
        "python",
        "fail",
        f"Python {version} is too old; the CLI needs 3.12+",
        fix="Install Python 3.12 or newer, then reinstall the CLI.",
        affects=["everything"],
    )


def _check_api_key(profile: str) -> Check:
    try:
        key = config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return _check(
            "api-key",
            "fail",
            "No API key found.",
            fix="Run 'aai login' (or set ASSEMBLYAI_API_KEY).",
            affects=["everything"],
        )
    # validate_key doubles as the connectivity probe: it makes one cheap authed call,
    # so a pass means the key is valid AND api.assemblyai.com is reachable.
    try:
        valid = client.validate_key(key)
    except CLIError as exc:
        return _check(
            "api-key",
            "fail",
            f"Could not reach AssemblyAI: {exc.message}",
            fix="Check your network/proxy and that api.assemblyai.com is reachable.",
            affects=["everything"],
        )
    if valid:
        return _check("api-key", "ok", "API key is valid and AssemblyAI is reachable.")
    return _check(
        "api-key",
        "fail",
        "API key was rejected (HTTP 401).",
        fix="Run 'aai login' with a valid key.",
        affects=["everything"],
    )


def check_ffmpeg() -> Check:
    # ffmpeg is ONLY used to stream non-WAV files or URLs (stream/agent), where it
    # decodes them to 16 kHz mono PCM on the fly. Plain `transcribe` (including
    # YouTube URLs) uploads the file to AssemblyAI and never invokes ffmpeg, so it is
    # not required for transcription.
    if shutil.which("ffmpeg"):
        return _check("ffmpeg", "ok", "found")
    return _check(
        "ffmpeg",
        "warn",
        (
            "ffmpeg not found. Only needed to stream non-WAV files or URLs; "
            "transcription (including YouTube) works without it, as does streaming a "
            "16 kHz mono WAV."
        ),
        fix="Install ffmpeg (macOS: brew install ffmpeg; Debian/Ubuntu: apt-get install ffmpeg).",
        affects=["stream/agent (non-WAV file or URL input)"],
    )


def _probe_input_devices() -> int:
    """Number of available microphone (input) devices. Raises if audio is unavailable."""
    sd = _sounddevice()
    devices = sd.query_devices()
    return sum(1 for device in devices if _input_channels(device) > 0)


def _sounddevice() -> _SoundDeviceModule:
    import sounddevice as module

    sd: _SoundDeviceModule = module
    return sd


def _input_channels(device: Mapping[str, object]) -> int:
    channels = device.get("max_input_channels")
    return channels if isinstance(channels, int) else 0


def check_audio() -> Check:
    affects = ["stream (microphone)", "agent"]
    try:
        inputs = _probe_input_devices()
    except ImportError:
        return _check(
            "audio",
            "warn",
            "sounddevice is not importable; the microphone can't be used.",
            fix="pip install --force-reinstall sounddevice",
            affects=affects,
        )
    except Exception as exc:  # noqa: BLE001 - any PortAudio/device failure is a soft warning
        return _check(
            "audio",
            "warn",
            f"audio system unavailable: {exc}",
            fix="On Linux install PortAudio: sudo apt-get install libportaudio2",
            affects=affects,
        )
    if inputs == 0:
        return _check(
            "audio",
            "warn",
            "No microphone (input device) found.",
            fix="Connect a microphone; live mic input is needed for stream/agent.",
            affects=affects,
        )
    return _check("audio", "ok", f"{inputs} microphone input device(s) available.")


def _check_coding_agent() -> Check:
    missing = [tool for tool in ("claude", "npx") if shutil.which(tool) is None]
    if not missing:
        return _check(
            "coding-agent",
            "ok",
            "claude and npx found; run 'aai setup install' to wire up the docs MCP + skills.",
        )
    return _check(
        "coding-agent",
        "warn",
        f"not found: {', '.join(missing)}.",
        fix=(
            "Install Claude Code (https://claude.com/claude-code) and Node.js, "
            "then run 'aai setup install'."
        ),
        affects=["aai setup install"],
    )


def render(data: DoctorResult) -> str:
    checks = data["checks"]
    lines = [output.heading("Environment check")]
    for c in checks:
        symbol, style = _SYMBOL.get(c["status"], (theme.SYMBOL_HINT, "aai.muted"))
        lines.append(
            f"  [{style}]{escape(symbol)}[/{style}] {escape(c['name'])} — {escape(c['detail'])}"
        )
        if c["fix"]:
            lines.append("      " + output.hint(f"fix: {escape(c['fix'])}"))
    if data["ok"]:
        lines.append("  " + output.success("Everything looks good."))
    else:
        failed = sum(1 for c in checks if c["status"] == "fail")
        noun = "problem" if failed == 1 else "problems"
        lines.append("  " + output.fail(f"{failed} {noun} found — see fixes above."))
    return "\n".join(lines)


@app.command(
    rich_help_panel=help_panels.SETUP,
    epilog=examples_epilog(
        [
            ("Check your environment is ready", "aai doctor"),
            ("Output results as JSON", "aai doctor --json"),
        ]
    ),
)
def doctor(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Check that your environment is ready to use AssemblyAI."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        checks = [
            check_python(),
            _check_api_key(profile),
            check_ffmpeg(),
            check_audio(),
            _check_coding_agent(),
        ]
        ok = not any(c["status"] == "fail" for c in checks)
        payload: DoctorResult = {"ok": ok, "checks": checks}
        output.emit(payload, render, json_mode=json_mode)
        if not ok:
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
