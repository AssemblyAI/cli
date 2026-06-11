#!/bin/sh
# Cut an AssemblyAI CLI release: tag the version from pyproject.toml and push the
# tag, which triggers .github/workflows/release.yml (builds the arm64 bottle,
# creates the GitHub Release, opens the formula PR).
#
#   ./scripts/cut_release.sh         # verify, confirm, then tag + push
#   ./scripts/cut_release.sh --yes   # skip the interactive confirmation
#   ./scripts/cut_release.sh -n      # dry run: verify only, don't tag or push
#
# Bump the version (pyproject.toml + aai_cli/__init__.py) and merge that PR
# BEFORE running this — the script tags whatever version main currently holds.
set -eu

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -y | --yes) ASSUME_YES=1 ;;
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
tag="v${version}"

# __version__ must match (the `version` command and tests read it).
init_version="$(grep -m1 '__version__' aai_cli/__init__.py | sed -E 's/.*"([^"]+)".*/\1/')"
[ "$init_version" = "$version" ] ||
  err "version mismatch: pyproject.toml=$version but aai_cli/__init__.py=$init_version."

# --- Safety gates ----------------------------------------------------------
[ -f .github/workflows/release.yml ] ||
  err "release.yml not found on this checkout — merge the release pipeline to main first."

branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || err "on branch '$branch'; releases are tagged from 'main'."

[ -z "$(git status --porcelain)" ] || err "working tree is dirty; commit or stash first."

info "Fetching origin..."
git fetch --quiet origin main

[ "$(git rev-parse main)" = "$(git rev-parse origin/main)" ] ||
  err "local main is not in sync with origin/main; pull/push so they match, then retry."

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
  err "tag ${tag} already exists locally."
fi
if [ -n "$(git ls-remote --tags origin "refs/tags/${tag}")" ]; then
  err "tag ${tag} already exists on origin."
fi

sha="$(git rev-parse --short HEAD)"
info "Ready to release ${tag} at ${sha} (main, in sync with origin)."

if [ "$DRY_RUN" -eq 1 ]; then
  info "Dry run: all checks passed. Not tagging or pushing."
  exit 0
fi

# --- Confirm, then tag + push ----------------------------------------------
if [ "$ASSUME_YES" -ne 1 ]; then
  printf 'Tag %s at %s and push to origin? [y/N] ' "$tag" "$sha"
  read -r reply
  case "$reply" in
    [yY] | [yY][eE][sS]) ;;
    *) err "aborted." ;;
  esac
fi

git tag -a "$tag" -m "Release ${tag}"
info "Created tag ${tag}. Pushing..."
git push origin "$tag"

info "Pushed ${tag}. release.yml is now building the bottle."
info "Next: watch it with 'gh run watch', then admin-merge the 'release/${tag}-formula' PR."
