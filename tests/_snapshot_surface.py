"""Shared pieces of the CLI-surface snapshot suite (help + error goldens).

The ``--help`` goldens are split into one module per command group
(``tests/test_snapshots_help_<group>.py``) so concurrent branches that touch
different commands regenerate *different* ``.ambr`` files instead of all
conflicting in a single snapshot file. ``HELP_GROUPS`` is the partition;
``tests/test_snapshots_help_groups.py`` guards that it stays complete and
disjoint, so a new top-level command fails loudly until it is assigned a group.
"""

from __future__ import annotations

import re

from syrupy.assertion import SnapshotAssertion
from typer.testing import CliRunner

from aai_cli.main import app
from tests._cli_tree import leaf_command_argvs

# Top-level command name -> snapshot module group, mirroring the help panels in
# aai_cli.main._COMMAND_ORDER (plus the hidden _update-check). The keys are the
# ``tests/test_snapshots_help_<group>.py`` module suffixes.
HELP_GROUPS: dict[str, frozenset[str]] = {
    "build": frozenset({"onboard", "init", "dev", "share", "deploy"}),
    "run": frozenset(
        {
            "transcribe",
            "t",  # hidden alias for transcribe
            "stream",
            "dictate",
            "agent",
            "speak",
            "llm",
            "clip",
            "dub",
            "caption",
            "eval",
            "webhooks",
        }
    ),
    "tools": frozenset({"doctor", "setup", "config", "update", "telemetry", "_update-check"}),
    "history": frozenset({"transcripts", "sessions"}),
    "account": frozenset(
        {"login", "logout", "whoami", "balance", "usage", "limits", "keys", "audit"}
    ),
}

_runner = CliRunner()

# Matches SGR (color/style) ANSI escape sequences.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


def normalize(text: str) -> str:
    """Strip color and Rich's right-padding so the goldens pin plain text.

    CI runners set FORCE_COLOR (which overrides NO_COLOR in Rich), so renders
    carry escape sequences that are absent locally; and Rich right-pads every
    line to the console width, which would make the goldens unreadable and
    fragile under whitespace-trimming tooling.
    """
    text = _ANSI_SGR.sub("", text)
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def help_group_argvs(group: str) -> list[list[str]]:
    """Leaf-command argvs whose top-level command belongs to ``group``."""
    return [argv for argv in leaf_command_argvs() if argv[0] in HELP_GROUPS[group]]


def assert_help_matches_snapshot(argv: list[str], snapshot: SnapshotAssertion) -> None:
    """Invoke ``<argv> --help`` and compare the normalized render to its golden."""
    result = _runner.invoke(app, [*argv, "--help"])
    assert result.exit_code == 0
    assert normalize(result.output) == snapshot
