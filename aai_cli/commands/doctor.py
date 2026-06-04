from __future__ import annotations

import shutil
import sys
from typing import TypedDict

import typer
from rich.markup import escape

from aai_cli import client, config, output
from aai_cli.context import AppState, resolve_profile, run_command
from aai_cli.errors import CLIError, NotAuthenticated

app = typer.Typer()


class Check(TypedDict):
    """One diagnostic: a named check, its status, what it affects, and how to fix it."""

    name: str
    status: str  # "ok" | "warn" | "fail" — only "fail" makes `doctor` exit non-zero
    affects: list[str]
    detail: str
    fix: str | None


# Status -> render style. "fail" is a blocker; "warn" is degraded-but-usable.
_STYLE = {"ok": "aai.success", "warn": "aai.warn", "fail": "aai.error"}


def _check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        return {"name": "python", "status": "ok", "affects": [], "detail": version, "fix": None}
    return {
        "name": "python",
        "status": "fail",
        "affects": ["everything"],
        "detail": f"Python {version} is too old; the CLI needs 3.10+",
        "fix": "Install Python 3.10 or newer, then reinstall the CLI.",
    }


def _check_api_key(profile: str) -> Check:
    affects = ["everything"]
    try:
        key = config.resolve_api_key(profile=profile)
    except NotAuthenticated:
        return {
            "name": "api-key",
            "status": "fail",
            "affects": affects,
            "detail": "No API key found.",
            "fix": "Run 'aai login' (or set ASSEMBLYAI_API_KEY).",
        }
    # validate_key doubles as the connectivity probe: it makes one cheap authed call,
    # so a pass means the key is valid AND api.assemblyai.com is reachable.
    try:
        valid = client.validate_key(key)
    except CLIError as exc:
        return {
            "name": "api-key",
            "status": "fail",
            "affects": affects,
            "detail": f"Could not reach AssemblyAI: {exc.message}",
            "fix": "Check your network/proxy and that api.assemblyai.com is reachable.",
        }
    if valid:
        return {
            "name": "api-key",
            "status": "ok",
            "affects": [],
            "detail": "API key is valid and AssemblyAI is reachable.",
            "fix": None,
        }
    return {
        "name": "api-key",
        "status": "fail",
        "affects": affects,
        "detail": "API key was rejected (HTTP 401).",
        "fix": "Run 'aai login' with a valid key.",
    }


def _check_ffmpeg() -> Check:
    # ffmpeg is ONLY used to stream non-WAV files or URLs (stream/agent), where it
    # decodes them to 16 kHz mono PCM on the fly. Plain `transcribe` (including
    # YouTube URLs) uploads the file to AssemblyAI and never invokes ffmpeg, so it is
    # not required for transcription.
    affects = ["stream/agent (non-WAV file or URL input)"]
    if shutil.which("ffmpeg"):
        return {"name": "ffmpeg", "status": "ok", "affects": [], "detail": "found", "fix": None}
    return {
        "name": "ffmpeg",
        "status": "warn",
        "affects": affects,
        "detail": (
            "ffmpeg not found. Only needed to stream non-WAV files or URLs; "
            "transcription (including YouTube) works without it, as does streaming a "
            "16 kHz mono WAV."
        ),
        "fix": "Install ffmpeg (macOS: brew install ffmpeg; Debian/Ubuntu: apt-get install ffmpeg).",
    }


def _probe_input_devices() -> int:
    """Number of available microphone (input) devices. Raises if audio is unavailable."""
    import sounddevice as sd

    devices = sd.query_devices()
    return sum(1 for d in devices if d.get("max_input_channels", 0) > 0)


def _check_audio() -> Check:
    affects = ["stream (microphone)", "agent"]
    try:
        inputs = _probe_input_devices()
    except ImportError:
        return {
            "name": "audio",
            "status": "warn",
            "affects": affects,
            "detail": "sounddevice is not importable; the microphone can't be used.",
            "fix": "pip install --force-reinstall sounddevice",
        }
    except Exception as exc:  # noqa: BLE001 - any PortAudio/device failure is a soft warning
        return {
            "name": "audio",
            "status": "warn",
            "affects": affects,
            "detail": f"audio system unavailable: {exc}",
            "fix": "On Linux install PortAudio: sudo apt-get install libportaudio2",
        }
    if inputs == 0:
        return {
            "name": "audio",
            "status": "warn",
            "affects": affects,
            "detail": "No microphone (input device) found.",
            "fix": "Connect a microphone; live mic input is needed for stream/agent.",
        }
    return {
        "name": "audio",
        "status": "ok",
        "affects": [],
        "detail": f"{inputs} microphone input device(s) available.",
        "fix": None,
    }


def _check_coding_agent() -> Check:
    affects = ["aai claude install"]
    missing = [tool for tool in ("claude", "npx") if shutil.which(tool) is None]
    if not missing:
        return {
            "name": "coding-agent",
            "status": "ok",
            "affects": [],
            "detail": "claude and npx found.",
            "fix": None,
        }
    return {
        "name": "coding-agent",
        "status": "warn",
        "affects": affects,
        "detail": f"not found: {', '.join(missing)}.",
        "fix": "Install Claude Code (https://claude.com/claude-code) and Node.js to wire up docs.",
    }


def _render(data: dict[str, object]) -> str:
    checks: list[Check] = data["checks"]  # type: ignore[assignment]
    lines = ["[aai.heading]AssemblyAI environment check:[/aai.heading]"]
    for c in checks:
        style = _STYLE.get(c["status"], "aai.muted")
        lines.append(
            f"  {escape(c['name'])}: "
            f"[{style}]{escape(c['status'])}[/{style}] — {escape(c['detail'])}"
        )
        if c["fix"]:
            lines.append(f"      [aai.muted]fix:[/aai.muted] {escape(c['fix'])}")
    if data["ok"]:
        lines.append("  [aai.success]Ready.[/aai.success]")
    else:
        lines.append("  [aai.error]Problems found — see fixes above.[/aai.error]")
    return "\n".join(lines)


@app.command()
def doctor(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Check that your environment is ready to use AssemblyAI."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = resolve_profile(state)
        checks = [
            _check_python(),
            _check_api_key(profile),
            _check_ffmpeg(),
            _check_audio(),
            _check_coding_agent(),
        ]
        ok = not any(c["status"] == "fail" for c in checks)
        output.emit({"ok": ok, "checks": checks}, _render, json_mode=json_mode)
        if not ok:
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
