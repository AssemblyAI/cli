#!/bin/sh
# Bump the AssemblyAI CLI version by one minor increment (X.Y.Z -> X.(Y+1).0),
# updating the two files that must stay in lock-step: pyproject.toml and
# aai_cli/__init__.py. It does NOT commit, tag, or push.
#
#   ./scripts/bump_minor.sh         # bump and write both files
#   ./scripts/bump_minor.sh -n      # dry run: print the new version, write nothing
#
# Typical flow: run this on a branch, commit the change, open + merge the PR,
# then run ./scripts/cut_release.sh on main to tag the new version.
set -eu

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -n | --dry-run) DRY_RUN=1 ;;
    -h | --help)
      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      printf 'unknown argument: %s (try --help)\n' "$arg" >&2
      exit 2
      ;;
  esac
done

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
err() {
  printf '\033[1;31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

# Run from the repo root so the relative paths below resolve regardless of CWD.
root="$(git rev-parse --show-toplevel)" || err "not inside a git repository."
cd "$root"

# --- Single source of truth: the version in pyproject.toml -----------------
version="$(grep -m1 '^version = ' pyproject.toml | sed -E 's/^version = "([^"]+)".*/\1/')"
[ -n "$version" ] || err "could not read version from pyproject.toml."

# __version__ must already match, or the two files would diverge on the bump.
init_version="$(grep -m1 '__version__' aai_cli/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
[ "$init_version" = "$version" ] ||
  err "version mismatch: pyproject.toml=$version but aai_cli/__init__.py=$init_version."

# --- Compute the minor bump (X.Y.Z -> X.(Y+1).0) ---------------------------
echo "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' ||
  err "version '$version' is not a plain MAJOR.MINOR.PATCH triple; bump it by hand."

major="$(echo "$version" | cut -d. -f1)"
minor="$(echo "$version" | cut -d. -f2)"
new_version="${major}.$((minor + 1)).0"
info "Bumping ${version} -> ${new_version}."

if [ "$DRY_RUN" -eq 1 ]; then
  info "Dry run: not writing any files."
  exit 0
fi

# --- Rewrite both files (portable in-place edit via temp file) -------------
replace_in_file() {
  # $1 file, $2 sed expression
  tmp="$(mktemp)"
  sed -E "$2" "$1" >"$tmp"
  mv "$tmp" "$1"
}

replace_in_file pyproject.toml "s/^version = \"${version}\"/version = \"${new_version}\"/"
replace_in_file aai_cli/__init__.py "s/__version__ = \"${version}\"/__version__ = \"${new_version}\"/"

info "Updated pyproject.toml and aai_cli/__init__.py to ${new_version}."
info "Next: commit the change, open + merge the PR, then run ./scripts/cut_release.sh on main."
