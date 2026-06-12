"""Guard: the per-module help-snapshot split must cover the whole command tree.

The ``--help`` goldens are split across ``tests/test_snapshots_help_<group>.py``
so concurrent branches touch different ``.ambr`` files. That split only stays
safe if the groups partition the live Typer tree — every top-level command in
exactly one group. A new command fails here until it is assigned to a group in
``HELP_GROUPS`` (tests/_snapshot_surface.py), which preserves the old single
file's property that every command is automatically required to have a help
snapshot.
"""

from __future__ import annotations

from collections import Counter

from tests._cli_tree import leaf_command_argvs
from tests._snapshot_surface import HELP_GROUPS


def test_help_groups_partition_the_command_tree():
    grouped = Counter(name for names in HELP_GROUPS.values() for name in names)
    duplicated = sorted(name for name, count in grouped.items() if count > 1)
    assert duplicated == []  # each command's goldens live in exactly one module
    live = {argv[0] for argv in leaf_command_argvs()}
    assert set(grouped) == live  # a new top-level command must be assigned a group
