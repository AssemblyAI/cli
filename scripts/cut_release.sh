#!/bin/sh
# Cut an AssemblyAI CLI release: tag the version and push the tag, which triggers
# .github/workflows/release.yml (builds the arm64 bottle, creates the GitHub
# Release, opens the formula PR).
#
# With hatch-vcs the git tag IS the version — there is no version file to bump
# or version-bump PR to merge first. By default the script tags the next patch
# above the latest vX.Y.Z tag; pass an explicit version to override.
#
#   ./scripts/cut_release.sh         # next patch above latest tag; confirm, tag + push
#   ./scripts/cut_release.sh 0.2.0   # tag an explicit version instead
#   ./scripts/cut_release.sh --yes   # skip the interactive confirmation
#   ./scripts/cut_release.sh -n      # dry run: verify only, don't tag or push
set -eu

ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    -y | --yes) ASSUME_YES=1 ;;
    -n | --dry-run) DRY_RUN=1 ;;
    -h | --help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    [0-9]*.[0-9]*.[0-9]*) EXPLICIT_VERSION="$arg" ;;
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

# --- Resolve the version to tag --------------------------------------------
# With hatch-vcs the git tag IS the version; there is no file to read. Default
# to the next patch above the latest vX.Y.Z tag; an explicit arg overrides.
latest="$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n1)"

if [ -n "${EXPLICIT_VERSION:-}" ]; then
  version="$EXPLICIT_VERSION"
else
  # Auto-bump needs a tag to bump from; an explicit version does not.
  [ -n "$latest" ] || err "no existing vX.Y.Z tag found; pass an explicit version."
  base="${latest#v}"
  major="$(echo "$base" | cut -d. -f1)"
  minor="$(echo "$base" | cut -d. -f2)"
  patch="$(echo "$base" | cut -d. -f3)"
  version="${major}.${minor}.$((patch + 1))"
fi
echo "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$' ||
  err "version '$version' is not a plain MAJOR.MINOR.PATCH triple."
tag="v${version}"
if [ -n "$latest" ]; then
  info "Latest tag ${latest}; releasing ${tag}."
else
  info "No prior tags; releasing ${tag}."
fi

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
