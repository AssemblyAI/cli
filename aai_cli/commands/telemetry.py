"""`aai telemetry` — inspect or change anonymous usage telemetry.

The collection itself lives in ``aai_cli/telemetry.py``; this command is the
user-facing consent surface: see what would be sent and turn it off (or back
on) persistently. The env kill-switches (``AAI_TELEMETRY_DISABLED=1``,
``DO_NOT_TRACK=1``) always win over the persisted choice.
"""

from __future__ import annotations

import typer

from aai_cli import config, options, output, telemetry
from aai_cli.context import AppState, run_command
from aai_cli.help_text import examples_epilog

app = typer.Typer(help="Anonymous usage telemetry: status, enable, disable.")


def _consent_label() -> str:
    return "granted" if telemetry.consent_granted() else "denied"


@app.command(
    epilog=examples_epilog(
        [
            ("Show whether telemetry is active", "aai telemetry status"),
            ("As JSON for scripting", "aai telemetry status --json"),
        ]
    )
)
def status(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Show whether anonymous usage telemetry is active, and why."""

    def body(_state: AppState, json_mode: bool) -> None:
        data: dict[str, object] = {
            "enabled": telemetry.is_enabled(),
            "consent": _consent_label(),
            "token_configured": bool(telemetry.client_token()),
        }

        def render(d: dict[str, object]) -> object:
            state_line = (
                output.success("Telemetry is enabled.")
                if d["enabled"]
                else output.muted("Telemetry is disabled.")
            )
            detail = output.muted(
                f"Consent: {d['consent']}. Intake token configured: "
                f"{'yes' if d['token_configured'] else 'no'}."
            )
            hint = output.hint(
                "Opt out any time: 'aai telemetry disable' or AAI_TELEMETRY_DISABLED=1."
            )
            return output.stack(state_line, detail, hint)

        output.emit(data, render, json_mode=json_mode)

    run_command(ctx, body, json=json_out)


@app.command(
    epilog=examples_epilog([("Re-enable telemetry", "aai telemetry enable")]),
)
def enable(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Re-enable anonymous usage telemetry for this machine."""

    def body(_state: AppState, json_mode: bool) -> None:
        config.set_telemetry_enabled(enabled=True)
        output.emit(
            {"telemetry_enabled": True},
            lambda _d: output.success("Telemetry enabled."),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)


@app.command(
    hidden=True,
    epilog=examples_epilog(
        [("Internal plumbing, spawned by the CLI itself", "aai telemetry flush '<payload-json>'")]
    ),
)
def flush(
    payload: str = typer.Argument(..., help="Serialized telemetry payload (internal)."),
) -> None:
    """Deliver one serialized telemetry event to the intake (internal).

    This is the detached flusher `telemetry.dispatch` spawns so user commands never
    wait on the network — an explicit, reviewable entry point rather than inline
    code. Hidden from help; runs with stdio discarded, so it neither needs nor
    produces output.
    """
    telemetry.flush_payload(payload)


@app.command(
    epilog=examples_epilog([("Opt out of telemetry", "aai telemetry disable")]),
)
def disable(
    ctx: typer.Context,
    json_out: bool = options.json_option(),
) -> None:
    """Opt out of anonymous usage telemetry for this machine."""

    def body(_state: AppState, json_mode: bool) -> None:
        config.set_telemetry_enabled(enabled=False)
        output.emit(
            {"telemetry_enabled": False},
            lambda _d: output.success("Telemetry disabled."),
            json_mode=json_mode,
        )

    run_command(ctx, body, json=json_out)
