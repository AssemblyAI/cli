"""Stream-discipline sweep over every leaf command.

The repo-wide invariant is *"Errors → stderr, data → stdout"* (root ``AGENTS.md``)
— it's what keeps ``assembly … --json | next-tool`` pipeline-safe. Individual
commands assert their own happy-path JSON shape, but nothing swept the contract
across *all* of them, so a command that leaked a human-readable line onto stdout
in ``--json`` mode (or printed an error payload to stdout) would pass the whole
gate. This walks the live Typer tree and pins the contract for every leaf.

The trigger is an **unknown flag**: it fails during Click's argument parsing,
before the command body, before any credential resolution or network — so it is
the one error path that is uniform across all commands *and* deterministic under
the suite's ``--disable-socket`` (no command-specific required-arg knowledge, and
no risk of an interactive command like ``login``/``stream`` blocking on a browser
or a mic). Click 8.2+ keeps ``result.stdout`` and ``result.stderr`` as separate
streams on the ``CliRunner`` ``Result``, so the split is observable directly.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aai_cli.main import app
from tests._cli_tree import leaf_command_argvs

runner = CliRunner()

# A flag no command defines, so it always trips Click's "No such option" parse
# error rather than reaching a command body.
UNKNOWN_FLAG = "--__definitely-not-a-real-flag__"


def _json_lines(stream: str) -> list[object]:
    """Parse a stream as NDJSON, asserting every line is valid JSON.

    A human-prose leak (a Rich-rendered error, a bare status line) is exactly a
    line that fails ``json.loads`` — so this raises rather than silently skipping.
    """
    return [json.loads(line) for line in stream.splitlines()]


@pytest.mark.parametrize("path", leaf_command_argvs(), ids=lambda p: " ".join(p))
def test_json_mode_keeps_stdout_clean_and_error_on_stderr(path: list[str]) -> None:
    result = runner.invoke(app, [*path, UNKNOWN_FLAG, "--json"])

    # Usage/parse error: the stable exit code for a bad flag (REFERENCE.md table).
    assert result.exit_code == 2
    # The whole point of the pipeline contract: a parse error puts *nothing* on
    # stdout, so a downstream consumer never sees a partial/garbage record.
    assert result.stdout == ""
    # The error rides stderr as the uniform JSON envelope — machine-readable, and
    # every emitted line parses (a human-prose leak onto stderr fails here too).
    objs = _json_lines(result.stderr)
    assert {"error": {"type": "usage_error", "message": "No such option: " + UNKNOWN_FLAG}} in objs


@pytest.mark.parametrize("path", leaf_command_argvs(), ids=lambda p: " ".join(p))
def test_human_mode_routes_errors_to_stderr(path: list[str]) -> None:
    # The same contract without --json: a human error still belongs on stderr, never
    # stdout, so `assembly … -o text > out` keeps the error out of the data file.
    result = runner.invoke(app, [*path, UNKNOWN_FLAG])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "No such option" in result.stderr
