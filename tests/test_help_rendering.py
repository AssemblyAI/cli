"""Rendering guards for the help screens and Click error formatting.

These pin the two patches main.py applies to Typer's rich rendering: flag-name
columns must never be clipped to an unreadable "--end-of-turn-c…" at a standard
80-column terminal, and unknown-flag suggestions must not leak a tuple repr.
"""

import json
import re
import sys

import pytest
from typer.testing import CliRunner

from aai_cli.main import app
from tests._cli_tree import leaf_command_argvs

runner = CliRunner()

# Matches SGR (color/style) ANSI escape sequences. CI runners force color on (Rich
# enables it under GITHUB_ACTIONS), which interleaves style codes mid-message —
# e.g. the option highlighter styles "--zzqq" inside "No such option: --zzqq" —
# so every text assertion here runs on the color-free render.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")

# A long-option token cut off by Rich's ellipsis overflow, e.g. "--end-of-turn-c…".
_CLIPPED_FLAG = re.compile(r"--[\w-]*…")


def _plain(text: str) -> str:
    return _ANSI_SGR.sub("", text)


@pytest.mark.parametrize("argv", leaf_command_argvs(), ids=" ".join)
def test_no_flag_name_is_clipped_at_80_columns(argv):
    result = runner.invoke(app, [*argv, "--help"], env={"COLUMNS": "80"})
    assert result.exit_code == 0
    plain = _plain(result.output)
    assert not _CLIPPED_FLAG.search(plain), _CLIPPED_FLAG.search(plain)


def test_unknown_flag_suggestion_renders_clean():
    # Vendored Click formats this as a stringified 1-tuple ("('(Possible options:
    # --json)',)"); main.py folds the suggestion into the message instead.
    result = runner.invoke(app, ["transcribe", "x.wav", "--jsno"])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "(Possible options: --json)" in plain
    assert "',)" not in plain


def test_unknown_flag_without_suggestion_renders_plain():
    result = runner.invoke(app, ["transcribe", "x.wav", "--zzqq"])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "No such option: --zzqq" in plain
    assert "Possible options" not in plain
    # An unknown flag that isn't a misplaced global gets no placement hint either.
    assert "global flag" not in plain


def _json_error(result):
    """Parse the single {"error": …} envelope a JSON-mode failure emits on stderr."""
    line = next(line for line in result.output.splitlines() if line.startswith("{"))
    return json.loads(line)["error"]


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["transcripts", "get", "--json"], "Missing argument 'TRANSCRIPT_ID'"),
        (["llm", "hi", "--max-tokens", "abc", "--json"], "not a valid integer"),
    ],
    ids=["missing-argument", "bad-option-value"],
)
def test_parse_error_with_json_emits_error_envelope(argv, fragment):
    # Click-level parse errors must honor --json like root-callback failures do:
    # a machine-readable {"error": …} envelope on stderr, exit code still 2.
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    error = _json_error(result)
    assert error["type"] == "usage_error"
    assert fragment in error["message"]
    # The human Usage/panel chrome is replaced by the envelope, not added to it.
    assert "Usage:" not in result.output


def test_parse_error_without_json_keeps_human_panel():
    result = runner.invoke(app, ["transcripts", "get"])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "Usage: assembly transcripts get" in plain
    assert "Missing argument 'TRANSCRIPT_ID'" in plain
    assert '{"error"' not in plain


def test_root_level_unknown_flag_gets_no_placement_hint():
    # Only the known misplaced flags earn a hint; a root-level typo stays a plain error.
    result = runner.invoke(app, ["--zzqq", "transcribe", "x.wav"])
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "No such option: --zzqq" in plain
    assert "Pass --json after the subcommand" not in plain


def test_root_level_json_flag_points_at_subcommand_placement():
    # `--json` is a per-command flag; at root level the old similarity guess
    # ("Possible options: --version") was actively misleading. And since the user
    # asked for JSON, the error itself arrives as the JSON envelope.
    result = runner.invoke(app, ["--json", "transcribe", "x.wav"])
    assert result.exit_code == 2
    error = _json_error(result)
    assert error["type"] == "usage_error"
    assert "No such option: --json" in error["message"]
    assert "Pass --json after the subcommand: assembly <command> --json" in error["message"]
    assert "Possible options" not in error["message"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["doctor", "-q"], "assembly -q doctor"),
        (["speak", "hi", "--sandbox"], "assembly --sandbox speak"),
        # --version is declared on the root callback like any other global flag; the
        # hint set is derived from those declarations, so it gets the hint too (the
        # old hand-maintained list missed it).
        (["doctor", "--version"], "assembly --version doctor"),
    ],
    ids=["-q", "--sandbox", "--version"],
)
def test_global_flag_on_subcommand_points_at_root_placement(argv, expected):
    result = runner.invoke(app, argv, env={"COLUMNS": "300"})
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert f"No such option: {argv[-1]}" in plain
    assert "This is a global flag; pass it before the subcommand" in plain
    assert expected in plain


def test_version_command_suggests_version_flag():
    # No `version` subcommand exists; the closest-match engine used to suggest the
    # unrelated 'sessions'. Point at the real spelling instead.
    result = runner.invoke(app, ["version"], env={"COLUMNS": "300"})
    assert result.exit_code == 2
    plain = _plain(result.output)
    assert "Did you mean 'assembly --version'?" in plain
    assert "sessions" not in plain


def test_misplaced_flag_hint_without_context_is_none():
    from typer._click.exceptions import NoSuchOption

    from aai_cli.main import _misplaced_flag_hint

    assert _misplaced_flag_hint(NoSuchOption("--json")) is None


def test_click_error_without_context_falls_back_to_argv(monkeypatch, capsys):
    # A ClickException raised without a context has no stashed token list; the
    # formatter then sniffs the real process argv for the JSON opt-in.
    from typer._click.exceptions import ClickException

    from aai_cli.main import _format_click_error_fixed

    monkeypatch.setattr(sys, "argv", ["assembly", "--json"])
    _format_click_error_fixed(ClickException("boom"))
    captured = capsys.readouterr().err
    assert json.loads(captured) == {"error": {"type": "usage_error", "message": "boom"}}

    monkeypatch.setattr(sys, "argv", ["assembly"])
    _format_click_error_fixed(ClickException("boom"))
    captured = capsys.readouterr().err
    assert "boom" in captured
    assert '{"error"' not in captured


def test_noclip_table_pins_leading_columns_and_passes_row_args_through():
    from aai_cli.main import _NoClipTable

    table = _NoClipTable()
    table.add_row("--flag", "META", "help text")
    # Only the trailing two (prose) columns may wrap/shrink.
    assert [c.no_wrap for c in table.columns] == [True, False, False]
    # add_row keyword defaults must match rich's (end_section=False unless asked).
    assert table.rows[-1].end_section is False
    table.add_row("--other", "META", "more", end_section=True)
    assert table.rows[-1].end_section is True
