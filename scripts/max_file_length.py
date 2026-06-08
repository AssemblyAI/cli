from __future__ import annotations

import sys
from pathlib import Path

# Files longer than this are hard for humans and AI coding agents alike to hold in
# context and navigate; split them into focused modules instead. Raising this limit
# should be a deliberate, reviewed decision — bump the number here, don't special-case.
MAX_LINES = 500

# First-party Python that the team reads and edits. Generated trees (docs/, dist/,
# build artifacts) and third-party code are out of scope.
ROOTS = ("aai_cli", "tests", "scripts")


def _line_count(path: Path) -> int:
    # Count newlines; a trailing line without a newline still counts (+1) so the
    # number matches `wc -l` on POSIX-clean files and never undercounts.
    with path.open("rb") as handle:
        data = handle.read()
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


def _offenders() -> list[tuple[Path, int]]:
    repo_root = Path(__file__).resolve().parent.parent
    found: list[tuple[Path, int]] = []
    for root in ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            count = _line_count(path)
            if count > MAX_LINES:
                found.append((path.relative_to(repo_root), count))
    return found


def main() -> int:
    offenders = _offenders()
    if not offenders:
        sys.stdout.write(f"All Python files are within {MAX_LINES} lines.\n")
        return 0
    sys.stdout.write(f"Files over the {MAX_LINES}-line limit (split them into smaller modules):\n")
    for path, count in offenders:
        sys.stdout.write(f"  {path}: {count} lines\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
