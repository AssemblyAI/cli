"""Shared pieces of the CLI-surface snapshot suite (help + error goldens).

The ``--help`` goldens are split into one module per command group
(``tests/test_snapshots_help_<group>.py``) so concurrent branches that touch
different commands regenerate *different* ``.ambr`` files instead of all
conflicting in a single snapshot file. ``HELP_GROUPS`` is the partition,
**derived** from each command module's ``SPEC.panel`` declaration (see
``aai_cli.command_registry``) so a new command is assigned a group by the same
declaration that registers it — no parallel dict to keep in sync.
``tests/test_snapshots_help_groups.py`` still guards that the derived partition
matches the live Typer tree, so a misdeclared ``SPEC`` fails loudly.
"""

from __future__ import annotations

import re

from syrupy.assertion import SnapshotAssertion
from typer.testing import CliRunner

from aai_cli import command_registry, help_panels
from aai_cli.main import app
from tests._cli_tree import leaf_command_argvs

# Help panel -> snapshot module group (the ``tests/test_snapshots_help_<group>.py``
# module suffix). A brand-new panel must be mapped here before its commands ship.
PANEL_TO_GROUP: dict[str, str] = {
    help_panels.QUICK_START: "build",
    help_panels.CODE: "run",
    help_panels.BUILD: "build",
    help_panels.TRANSCRIPTION: "run",
    help_panels.SETUP: "tools",
    help_panels.HISTORY: "history",
    help_panels.ACCOUNT: "account",
}


def _derive_help_groups() -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {group: set() for group in PANEL_TO_GROUP.values()}
    for registered in command_registry.discover():
        groups[PANEL_TO_GROUP[registered.spec.panel]].update(registered.spec.commands)
    # The hidden _update-check is registered directly in main.py, not via a SPEC.
    groups["tools"].add("_update-check")
    return {group: frozenset(names) for group, names in groups.items()}


# Top-level command name -> snapshot module group, derived from the registry.
HELP_GROUPS: dict[str, frozenset[str]] = _derive_help_groups()

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
