from __future__ import annotations

from pathlib import Path

import typer
from assemblyai.streaming.v3 import SpeechModel
from rich.markup import escape

from aai_cli import client, code_gen, output
from aai_cli.agent.session import DEFAULT_GREETING, DEFAULT_PROMPT
from aai_cli.agent.voices import DEFAULT_VOICE
from aai_cli.context import AppState, run_command
from aai_cli.errors import CLIError
from aai_cli.streaming.sources import TARGET_RATE

app = typer.Typer(
    help="Scaffold runnable AssemblyAI starter scripts.",
    no_args_is_help=True,
)

SAMPLES = ("transcribe", "stream", "agent")


def _generate(name: str) -> str:
    """Render a starter script via the same generator behind `--show-code`."""
    if name == "transcribe":
        return code_gen.transcribe({}, client.SAMPLE_AUDIO_URL)
    if name == "stream":
        return code_gen.stream(
            {
                "sample_rate": TARGET_RATE,
                "format_turns": True,
                "speech_model": SpeechModel.u3_rt_pro,
            }
        )
    return code_gen.agent(DEFAULT_VOICE, DEFAULT_PROMPT, DEFAULT_GREETING)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List available sample scripts."""

    def body(_state: AppState, json_mode: bool) -> None:
        output.emit(
            list(SAMPLES),
            lambda d: "Available samples:\n" + "\n".join(f"  - {n}" for n in d),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command()
def create(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Sample name."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing sample file."),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """Scaffold a runnable starter script (reads ASSEMBLYAI_API_KEY from the environment)."""

    def body(_state: AppState, json_mode: bool) -> None:
        if name not in SAMPLES:
            raise CLIError(
                f"Unknown sample '{name}'. Try: {', '.join(SAMPLES)}.",
                error_type="unknown_sample",
                exit_code=1,
            )
        target_dir = Path.cwd() / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{name}.py"
        if target.exists() and not force:
            raise CLIError(
                f"{target} already exists. Delete it or pass --force to overwrite.",
                error_type="file_exists",
                exit_code=1,
            )
        target.write_text(_generate(name))

        output.emit(
            {"created": str(target)},
            lambda d: (
                f"Created {escape(d['created'])}\n"
                f'Set your key (export ASSEMBLYAI_API_KEY="…"), then run: '
                f"python {escape(d['created'])}"
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
