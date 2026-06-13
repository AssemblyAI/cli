"""Golden snapshot pinning the exact `assembly --help` render (the root screen).

The root help lists every top-level command, so *any* new command changes it.
It lives in its own snapshot module — separate from the per-group goldens — so
that churn is confined to one trivially-regenerable ``.ambr`` file instead of
conflicting inside a command group's goldens. It also pins the derived command
ordering and panel layout (see ``aai_cli.command_registry``); refresh with::

    uv run pytest tests/test_snapshots_help_root.py --snapshot-update
"""

from __future__ import annotations

import pytest

from tests._snapshot_surface import assert_help_matches_snapshot

pytestmark = pytest.mark.usefixtures("fixed_render_size")


def test_root_help_matches_snapshot(snapshot):
    assert_help_matches_snapshot([], snapshot)
