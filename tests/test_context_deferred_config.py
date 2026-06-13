"""`run_command`'s handling of a corrupt-config error the root callback deferred.

Split out of test_context.py to keep modules under the 500-line gate. When config.toml
is unreadable, the callback can't resolve the profile's stored env; it defers the error
so `assembly config path` (which reports a contents-independent location) can still run,
while every other command re-raises it here.
"""

import typer
from typer.testing import CliRunner

from aai_cli.app.context import AppState, run_command
from aai_cli.core.errors import CLIError

runner = CliRunner()


def _make_app_with_deferred(body, deferred, **run_options):
    app = typer.Typer()

    @app.callback()
    def cb(ctx: typer.Context):
        ctx.obj = AppState(deferred_config_error=deferred)

    @app.command()
    def go(ctx: typer.Context):
        run_command(ctx, body, **run_options)

    return app


def test_run_command_reraises_deferred_config_error_for_normal_command():
    # A corrupt-config error the root callback deferred surfaces here for ordinary
    # commands — they never run their body against an unreadable config.
    deferred = CLIError("bad toml", error_type="invalid_config", exit_code=2)
    ran = {}

    result = runner.invoke(
        _make_app_with_deferred(lambda *_: ran.setdefault("body", True), deferred), ["go"]
    )
    assert result.exit_code == 2
    assert "bad toml" in result.output
    assert "body" not in ran  # body never ran


def test_run_command_tolerates_deferred_config_error_when_opted_in():
    # `config path` opts in: it runs despite the deferred error (and skips the
    # telemetry/update-check wrappers that would re-parse the broken config).
    deferred = CLIError("bad toml", error_type="invalid_config", exit_code=2)
    ran = {}

    result = runner.invoke(
        _make_app_with_deferred(
            lambda *_: ran.setdefault("body", True), deferred, tolerate_unreadable_config=True
        ),
        ["go"],
    )
    assert result.exit_code == 0
    assert ran.get("body") is True
    assert "bad toml" not in result.output
