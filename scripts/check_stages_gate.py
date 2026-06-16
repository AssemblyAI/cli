"""Keep the /check skill's gate-stage list in sync with scripts/check.sh.

scripts/check.sh is the single source of truth for what the verification gate
runs. This script derives the ordered stage labels from its `echo "==> ..."`
lines and verifies (or, with --write, regenerates) the managed block in the
/check skill, so the human-readable checklist agents read can never silently
drift from the code the way it had before this gate existed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECK_SH = REPO_ROOT / "scripts" / "check.sh"
SKILL = REPO_ROOT / ".claude" / "skills" / "check" / "SKILL.md"

BEGIN = "<!-- BEGIN GATE STAGES (generated from scripts/check.sh by scripts/check_stages_gate.py --write; do not edit by hand) -->"
END = "<!-- END GATE STAGES -->"

_STAGE_RE = re.compile(r'^\s*echo "==> (.+)"\s*$')
_BLOCK_RE = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)


def gate_stages() -> list[str]:
    """Return the ordered gate-stage labels declared in scripts/check.sh."""
    return [
        m.group(1)
        for line in CHECK_SH.read_text(encoding="utf-8").splitlines()
        if (m := _STAGE_RE.match(line))
    ]


def expected_block() -> str:
    """Render the managed block (markers + numbered stage list) for the skill."""
    lines = [BEGIN, ""]
    lines += [f"{i}. {label}" for i, label in enumerate(gate_stages(), start=1)]
    lines += ["", END]
    return "\n".join(lines)


def main() -> int:
    """Check (or, with --write, regenerate) the /check skill's gate-stage block."""
    parser = argparse.ArgumentParser(description="Sync the /check skill gate-stage list.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the managed block instead of just checking it",
    )
    args = parser.parse_args()

    rel = SKILL.relative_to(REPO_ROOT)
    text = SKILL.read_text(encoding="utf-8")
    if not _BLOCK_RE.search(text):
        sys.stdout.write(f"check_stages_gate: GATE STAGES markers not found in {rel}\n")
        return 1
    if not gate_stages():
        sys.stdout.write("check_stages_gate: no '==> ' stage labels found in scripts/check.sh\n")
        return 1

    updated = _BLOCK_RE.sub(lambda _: expected_block(), text)
    if args.write:
        if updated != text:
            SKILL.write_text(updated, encoding="utf-8")
            sys.stdout.write(f"Updated the gate-stage list in {rel}.\n")
        else:
            sys.stdout.write(f"The gate-stage list in {rel} is already up to date.\n")
        return 0

    if updated != text:
        sys.stdout.write(
            f"The gate-stage list in {rel} is out of sync with scripts/check.sh. "
            "Regenerate it:\n\n    uv run python scripts/check_stages_gate.py --write\n"
        )
        return 1
    sys.stdout.write(f"The gate-stage list in {rel} matches scripts/check.sh.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
