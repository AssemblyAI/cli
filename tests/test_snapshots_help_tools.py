"""Golden snapshots pinning the exact `--help` render of the tools commands.

One snapshot module per command group (see HELP_GROUPS in
tests/_snapshot_surface.py) keeps each group's goldens in its own .ambr file,
so concurrent changes to different commands don't conflict in one snapshot
file. A diff means the visible CLI changed; if intentional, refresh with::

    uv run pytest tests/test_snapshots_help_tools.py --snapshot-update

The argv list is derived from the live Typer tree filtered to this group, so a
new subcommand under one of these commands automatically requires a snapshot.
"""

from __future__ import annotations

import pytest

from tests._snapshot_surface import assert_help_matches_snapshot, help_group_argvs

pytestmark = pytest.mark.usefixtures("fixed_render_size")


@pytest.mark.parametrize("argv", help_group_argvs("tools"), ids=lambda a: "_".join(a))
def test_command_help_matches_snapshot(argv, snapshot):
    assert_help_matches_snapshot(argv, snapshot)
