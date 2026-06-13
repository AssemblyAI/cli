"""The root `--color` tri-state and `output.set_color_mode`."""

import os

import pytest
from typer.testing import CliRunner

from aai_cli import output
from aai_cli.choices import ColorMode
from aai_cli.main import app

runner = CliRunner()

_COLOR_VARS = ("NO_COLOR", "FORCE_COLOR")


@pytest.fixture(autouse=True)
def restore_color_state(monkeypatch):
    """set_color_mode swaps module-global consoles and writes env vars *during* the
    test (not via monkeypatch), so snapshot and restore both by hand. The vars are
    also cleared up front: CI exports FORCE_COLOR, which would skew the asserts.
    The update notice is disabled because a forced-terminal stderr console would
    otherwise convince it to spawn the real detached refresh process."""
    monkeypatch.setenv("AAI_NO_UPDATE_CHECK", "1")
    saved_env = {var: os.environ.pop(var, None) for var in _COLOR_VARS}
    saved_consoles = (output.console, output.error_console)
    yield
    for var, value in saved_env.items():
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value
    output.console, output.error_console = saved_consoles


def test_never_strips_color_and_sets_no_color_env():
    output.set_color_mode(ColorMode.never)
    assert os.environ["NO_COLOR"] == "1"
    assert "FORCE_COLOR" not in os.environ
    assert output.console.no_color is True
    assert output.error_console.no_color is True
    assert output.error_console.stderr is True  # stderr console stays on stderr


def test_always_forces_color_and_sets_force_color_env():
    os.environ["NO_COLOR"] = "1"  # an inherited NO_COLOR must lose to --color always
    output.set_color_mode(ColorMode.always)
    assert os.environ["FORCE_COLOR"] == "1"
    assert "NO_COLOR" not in os.environ
    assert output.console.is_terminal is True  # force_terminal even when captured
    assert output.error_console.is_terminal is True
    assert output.error_console.stderr is True


def test_auto_changes_nothing():
    before_out, before_err = output.console, output.error_console
    output.set_color_mode(ColorMode.auto)
    assert "NO_COLOR" not in os.environ
    assert "FORCE_COLOR" not in os.environ
    assert output.console is before_out  # the consoles aren't rebuilt
    assert output.error_console is before_err


def test_root_callback_wires_the_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(output, "set_color_mode", seen.append)
    result = runner.invoke(app, ["--color", "never", "config", "path"])
    assert result.exit_code == 0
    assert seen == [ColorMode.never]


def test_color_defaults_to_auto(monkeypatch):
    seen = []
    monkeypatch.setattr(output, "set_color_mode", seen.append)
    result = runner.invoke(app, ["config", "path"])
    assert result.exit_code == 0
    assert seen == [ColorMode.auto]


def test_color_rejects_unknown_value():
    result = runner.invoke(app, ["--color", "sometimes", "config", "path"])
    assert result.exit_code == 2
    assert "sometimes" in result.output


def test_never_yields_plain_output_end_to_end():
    # The user-visible effect: a forced-color invocation carries SGR escapes, a
    # --color never one does not (CliRunner captures off-TTY, so force via flag).
    plain = runner.invoke(app, ["--color", "never", "config", "list"])
    forced = runner.invoke(app, ["--color", "always", "config", "list"])
    assert "\x1b[" not in plain.output
    assert "\x1b[" in forced.output
