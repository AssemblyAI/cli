#!/usr/bin/env bash
# PreToolUse(Bash) gate: block `git commit` unless ./scripts/check.sh recorded a
# passing run for the *current* working tree. The full gate (mutation + 100% patch
# coverage tail) is what makes a commit safe here; a single-file `pytest` does not.
#
# Escape hatch for deliberate work-in-progress commits (e.g. before a multi-agent
# workflow, per the commit-wip-before-workflows practice): prefix the command with
#   AAI_ALLOW_COMMIT=1 git commit ...
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty')"

# Only police git commits; let everything else through immediately.
if ! printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+commit([[:space:]]|$)'; then
  exit 0
fi

# Explicit opt-out for intentional WIP commits.
if printf '%s' "$cmd" | grep -q 'AAI_ALLOW_COMMIT=1'; then
  exit 0
fi

# Fail open outside this repo or if the marker tool is missing, so the hook never
# wedges commits in an unrelated checkout.
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "$root" ] || [ ! -f "$root/scripts/gate_marker.py" ]; then
  exit 0
fi

if python3 "$root/scripts/gate_marker.py" check >/dev/null 2>&1; then
  exit 0
fi

cat >&2 <<'MSG'
Blocked: ./scripts/check.sh has not passed for the current working tree.

The full gate (incl. the mutation + 100% patch-coverage tail) must be green before
committing — a single-file `pytest` does not satisfy it. Run:

    ./scripts/check.sh

then commit again once it prints "All checks passed." Editing any file after the
gate passes re-requires it (the marker is keyed to the exact tree contents).

To commit work-in-progress on purpose, prefix the command:

    AAI_ALLOW_COMMIT=1 git commit ...
MSG
exit 2
