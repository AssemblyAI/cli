"""Rendering guards for the help screens and Click error formatting.

These pin the two patches main.py applies to Typer's rich rendering: flag-name
columns must never be clipped to an unreadable "--end-of-turn-c…" at a standard
80-column terminal, and unknown-flag suggestions must not leak a tuple repr.
"""

import re

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
