#!/usr/bin/env bash
# SessionStart hook: provision a Claude Code on the web container so the canonical
# gate (./scripts/check.sh) runs green here exactly as it does in CI and locally.
#
# Mirrors the deps CI installs (.github/workflows/ci.yml):
#   - system: libportaudio2 (sounddevice), ffmpeg (--sample stream sources), shellcheck
#   - node:   markdownlint-cli@0.45.0 (check.sh calls `markdownlint` directly)
#   - python: `uv sync` to materialize the locked dev environment up front
#
# Only runs in the remote (web) environment — local dev machines already have
# these and shouldn't be reprovisioned. Idempotent and non-interactive.
set -euo pipefail

# Web-only: skip entirely on local machines.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Pin markdownlint-cli to the version CI uses so lint results match exactly.
MARKDOWNLINT_VERSION="0.45.0"

log() { echo "[session-start] $*"; }

# 1. System packages (PortAudio + ffmpeg + shellcheck). apt-get is idempotent;
#    --no-install-recommends keeps the layer small. Skip the install if all three
#    are already present so resumed/cached containers don't re-run apt.
need_apt=0
command -v ffmpeg    >/dev/null 2>&1 || need_apt=1
command -v shellcheck >/dev/null 2>&1 || need_apt=1
ldconfig -p 2>/dev/null | grep -q portaudio || need_apt=1
if [ "$need_apt" = "1" ]; then
  log "installing system deps (libportaudio2, ffmpeg, shellcheck)"
  export DEBIAN_FRONTEND=noninteractive
  # `apt-get update` can exit non-zero when unrelated third-party PPAs are
  # unreachable; the main Ubuntu archive (which has these packages) still
  # refreshes, so tolerate that failure and let the install be the real signal.
  apt-get update -qq || log "apt-get update reported errors (likely unrelated PPAs); continuing"
  # Don't let a missing system package abort the whole session — a partially
  # provisioned container is still usable; check.sh self-skips shellcheck.
  apt-get install -y --no-install-recommends libportaudio2 ffmpeg shellcheck \
    || log "WARNING: apt-get install failed; some system deps may be missing"
else
  log "system deps already present"
fi

# 2. markdownlint CLI (Node) — check.sh invokes `markdownlint` as a global binary,
#    not through uv, so it must be on PATH.
if ! command -v markdownlint >/dev/null 2>&1; then
  log "installing markdownlint-cli@${MARKDOWNLINT_VERSION}"
  npm install -g "markdownlint-cli@${MARKDOWNLINT_VERSION}"
else
  log "markdownlint already present"
fi

# 3. Python environment — materialize the locked dev env so the first `uv run`
#    doesn't pay the full sync cost mid-task. `uv` syncs the default dev group.
log "syncing uv environment (locked dev group)"
uv sync

# 4. Keep the session branch current. Resumed web containers hold a clone frozen
#    at creation time, so two things can go stale: the branch's own remote tip
#    (pushes from another session/machine) and origin/main (which the diff-scoped
#    gates — diff-cover, mutation — compare against). Fast-forward to the remote
#    tip if it advanced, then merge origin/main if behind (the same semantics as
#    GitHub's "Update branch" button). Never force anything: a dirty tree skips
#    the update and a conflicting merge is aborted with a note, leaving the
#    resolution to the session.
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")
if [ "$branch" != "HEAD" ] && [ "$branch" != "main" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    log "working tree is dirty; skipping branch auto-update"
  elif git fetch origin main "$branch" 2>/dev/null || git fetch origin main 2>/dev/null; then
    if git rev-parse --verify --quiet "origin/$branch" >/dev/null; then
      if git merge-base --is-ancestor "HEAD" "origin/$branch" && [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$branch")" ]; then
        git merge --ff-only "origin/$branch"
        log "fast-forwarded $branch to its remote tip"
      fi
    fi
    behind=$(git rev-list --count "HEAD..origin/main" 2>/dev/null || echo 0)
    if [ "$behind" -gt 0 ]; then
      if git merge --no-edit origin/main; then
        log "merged origin/main into $branch (was $behind commit(s) behind)"
      else
        git merge --abort 2>/dev/null || true
        log "WARNING: origin/main conflicts with $branch; left unmerged — resolve with 'git merge origin/main'"
      fi
    else
      log "$branch is up to date with origin/main"
    fi
  else
    log "could not fetch origin; skipping branch auto-update"
  fi
fi

log "provisioning complete"
