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

# A long-option token cut off by Rich's ellipsis overflow, e.g. "--end-of-turn-c…".
_CLIPPED_FLAG = re.compile(r"--[\w-]*…")


@pytest.mark.parametrize("argv", leaf_command_argvs(), ids=" ".join)
def test_no_flag_name_is_clipped_at_80_columns(argv):
    result = runner.invoke(app, [*argv, "--help"], env={"COLUMNS": "80"})
    assert result.exit_code == 0
    assert not _CLIPPED_FLAG.search(result.output), _CLIPPED_FLAG.search(result.output)


def test_root_help_documents_authentication():
    # The shared "how do I authenticate / pick a backend" line: the env vars must be
    # discoverable from the CLI itself, not only from external docs.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ASSEMBLYAI_API_KEY" in result.output
    assert "AAI_ENV" in result.output


def test_unknown_flag_suggestion_renders_clean():
    # Vendored Click formats this as a stringified 1-tuple ("('(Possible options:
    # --json)',)"); main.py folds the suggestion into the message instead.
    result = runner.invoke(app, ["transcribe", "x.wav", "--jsno"])
    assert result.exit_code == 2
    assert "(Possible options: --json)" in result.output
    assert "',)" not in result.output


def test_unknown_flag_without_suggestion_renders_plain():
    result = runner.invoke(app, ["transcribe", "x.wav", "--zzqq"])
    assert result.exit_code == 2
    assert "No such option: --zzqq" in result.output
    assert "Possible options" not in result.output


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
