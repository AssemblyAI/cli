from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import typer
from rich.markup import escape

from assemblyai_cli import config, output
from assemblyai_cli.context import run_command
from assemblyai_cli.errors import CLIError

app = typer.Typer(help="Scaffold runnable AssemblyAI starter scripts.")

# template name -> (template resource filename, output filename)
TEMPLATES = {
    "transcribe": ("transcribe.py.tmpl", "transcribe.py"),
    "stream": ("stream.py.tmpl", "stream.py"),
}


@app.command(name="list")
def list_(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
) -> None:
    """List available sample templates."""

    def body(_state, json_mode: bool) -> None:
        names = sorted(TEMPLATES)
        output.emit(
            names,
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
    """Scaffold a runnable starter script with your API key injected."""

    def body(state, json_mode: bool) -> None:
        if name not in TEMPLATES:
            raise CLIError(
                f"Unknown sample '{name}'. Try: {', '.join(sorted(TEMPLATES))}.",
                error_type="unknown_sample",
                exit_code=1,
            )
        api_key = config.resolve_api_key(profile=state.profile)
        tmpl_file, out_file = TEMPLATES[name]
        template = resources.files("assemblyai_cli.templates").joinpath(tmpl_file).read_text()
        rendered = template.replace("{{API_KEY}}", api_key)

        target_dir = Path.cwd() / name
        target_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(target_dir, 0o700)
        target = target_dir / out_file
        if target.exists() and not force:
            raise CLIError(
                f"{target} already exists. Delete it or pass --force to overwrite.",
                error_type="file_exists",
                exit_code=1,
            )
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(rendered)
        os.chmod(target, 0o600)

        output.emit(
            {"created": str(target)},
            lambda d: (
                f"Created {escape(d['created'])}\n"
                f"[yellow]Note:[/yellow] this file contains your API key — do not commit it.\n"
                f"Run it with: python {escape(d['created'])}"
            ),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
