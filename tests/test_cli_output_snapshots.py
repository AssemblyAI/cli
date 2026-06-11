"""Golden/snapshot tests that pin the exact rendered CLI surface.

These lock the user-facing output we otherwise only spot-check with substring
asserts: every command's ``--help`` block (usage, options, and the Examples
epilog) and the human + JSON forms of errors (including the ``Suggestion:`` line).

A diff here means the visible CLI changed. If the change is intentional, refresh
the goldens with::

    python -m pytest tests/test_cli_output_snapshots.py --snapshot-update

The command list is derived from the live Typer tree, so a new command is
automatically required to have a help snapshot (mirroring the coverage guard in
``tests/test_help_examples_coverage.py``).
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from aai_cli import output
from aai_cli.errors import APIError, CLIError, NotAuthenticated, UsageError, auth_failure
from aai_cli.main import app
from tests._cli_tree import leaf_command_argvs

runner = CliRunner()


# Matches SGR (color/style) ANSI escape sequences.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _normalize(text: str) -> str:
    # Strip ANSI color codes first: CI runners set FORCE_COLOR (which overrides
    # NO_COLOR in Rich), so --help renders with escape sequences locally-absent.
    # The goldens pin plain text, so compare on the color-free render.
    text = _ANSI_SGR.sub("", text)
    # Rich right-pads every line to the console width; strip trailing spaces so the
    # goldens stay readable and survive whitespace-trimming tooling.
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


HELP_ARGVS = leaf_command_argvs()

# Representative errors: each auth/usage/api shape, plus one with no suggestion so
# the golden proves no stray "Suggestion:" line is emitted.
ERROR_CASES = {
    "api_no_suggestion": APIError("Transcription request failed: boom."),
    "not_authenticated": NotAuthenticated(),
    "plain_no_suggestion": CLIError("Something went wrong."),
    "rejected_key": auth_failure(),
    "usage_with_suggestion": UsageError(
        "Unknown voice 'nope'.",
        suggestion="Run 'assembly agent --list-voices' to see the options.",
    ),
}


@pytest.fixture(autouse=True)
def _fixed_render_size(monkeypatch):
    # Pin the render width so goldens are byte-identical across machines and CI.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "40")


@pytest.mark.parametrize("argv", HELP_ARGVS, ids=lambda a: "_".join(a))
def test_command_help_matches_snapshot(argv, snapshot):
    result = runner.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0
    assert _normalize(result.output) == snapshot


@pytest.mark.parametrize("case", sorted(ERROR_CASES), ids=lambda c: c)
def test_error_human_render_matches_snapshot(case, capsys, snapshot):
    output.emit_error(ERROR_CASES[case], json_mode=False)
    captured = capsys.readouterr()
    assert captured.out == ""  # errors never touch stdout (keeps pipelines clean)
    assert _normalize(captured.err) == snapshot


@pytest.mark.parametrize("case", sorted(ERROR_CASES), ids=lambda c: c)
def test_error_json_render_matches_snapshot(case, capsys, snapshot):
    output.emit_error(ERROR_CASES[case], json_mode=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == snapshot  # exact JSON line, including the suggestion key
