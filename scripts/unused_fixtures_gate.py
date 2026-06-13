from __future__ import annotations

import sys
from pathlib import Path

# Orphaned-test-artifact gate, modeled on the Deno toolchain's "every `.out` file must be
# referenced by a test" check (tools/lint.js). The unit suite runs under pytest-xdist
# (`-n auto`), which disables syrupy's own unused-snapshot reporting — each worker only
# sees a slice of the snapshots — so a renamed or deleted test can silently leave its
# whole snapshot file or a recorded API fixture behind to rot. This catches that
# statically and fast, with no extra test run.
#
# Two artifact kinds are checked:
#   * tests/__snapshots__/<name>.ambr  — syrupy names a snapshot file after its test
#     module, so each `.ambr` must have a matching tests/<name>.py.
#   * tests/fixtures/api/<name>.json   — replay fixtures are loaded by stem
#     (replay_fixtures.load_object("<name>")), so each must be referenced by name in
#     some test module (the loader module itself doesn't count).

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
SNAPSHOT_DIR = TESTS_DIR / "__snapshots__"
API_FIXTURE_DIR = TESTS_DIR / "fixtures" / "api"
# The fixture loader names every stem in its own docstring/paths, so it can't count as a
# real reference — only an actual test that loads the fixture should keep it alive.
LOADER_MODULE = "replay_fixtures.py"


def _orphaned_snapshots() -> list[Path]:
    """`.ambr` files whose owning test module no longer exists."""
    return [
        ambr.relative_to(REPO_ROOT)
        for ambr in sorted(SNAPSHOT_DIR.glob("*.ambr"))
        if not (TESTS_DIR / f"{ambr.stem}.py").exists()
    ]


def _test_sources() -> list[str]:
    """Bodies of every test module except the fixture loader."""
    return [
        path.read_text(encoding="utf-8")
        for path in sorted(TESTS_DIR.rglob("*.py"))
        if path.name != LOADER_MODULE
    ]


def _unreferenced_fixtures() -> list[Path]:
    """API fixtures whose stem is never named by a test module."""
    if not API_FIXTURE_DIR.exists():
        return []
    sources = _test_sources()
    return [
        fixture.relative_to(REPO_ROOT)
        for fixture in sorted(API_FIXTURE_DIR.glob("*.json"))
        if not any(fixture.stem in source for source in sources)
    ]


def main() -> int:
    snapshot_orphans = _orphaned_snapshots()
    fixture_orphans = _unreferenced_fixtures()
    if not snapshot_orphans and not fixture_orphans:
        sys.stdout.write("No orphaned snapshots or unreferenced fixtures.\n")
        return 0
    for path in snapshot_orphans:
        sys.stdout.write(f"Orphaned snapshot (no matching test module): {path}\n")
    for path in fixture_orphans:
        sys.stdout.write(f"Unreferenced API fixture (no test loads it): {path}\n")
    sys.stdout.write("Delete the dead artifact, or wire it back into a test.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
