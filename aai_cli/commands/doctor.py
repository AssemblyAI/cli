from __future__ import annotations

import typer

from aai_cli import command_registry, doctor_checks, environments, help_panels, options, output
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer()

SPEC = command_registry.CommandModuleSpec(
    panel=help_panels.SETUP,
    order=10,  # pragma: no mutate -- sparse rank; a +-1 shift is order-equivalent
    commands=("doctor",),
)


@app.command(
    rich_help_panel=help_panels.SETUP,
    epilog=examples_epilog(
        [
            ("Check your environment is ready", "assembly doctor"),
            ("Output results as JSON", "assembly doctor --json"),
        ]
    ),
)
def doctor(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Check that your environment is ready to use AssemblyAI."""

    def body(state: AppState, json_mode: bool) -> None:
        profile = state.resolve_profile()
        # Called through the module (not name-imported) so a monkeypatched check in
        # aai_cli.doctor_checks is seen here and by the onboarding wizard alike.
        checks = [
            doctor_checks.check_python(),
            doctor_checks.check_credentials(profile),
            doctor_checks.check_ffmpeg(),
            doctor_checks.check_audio(),
            doctor_checks.check_coding_agent(),
        ]
        ok = not any(c["status"] == "fail" for c in checks)
        payload: doctor_checks.DoctorResult = {
            "ok": ok,
            "profile": profile,
            "environment": environments.active().name,
            "checks": checks,
        }
        output.emit(payload, doctor_checks.render, json_mode=json_mode)
        if not ok:
            raise typer.Exit(code=1)

    run_command(ctx, body, json=json_out)
