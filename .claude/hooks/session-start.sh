#!/usr/bin/env bash
# SessionStart hook: provision a Claude Code on the web container so the canonical
# gate (./scripts/check.sh) runs green here exactly as it does in CI and locally.
#
# Mirrors the deps CI installs (.github/workflows/ci.yml); shared tool pins live
# in scripts/gate_tool_pins.sh:
#   - system: libportaudio2 (sounddevice), ffmpeg (--sample stream sources), shellcheck
#   - node:   markdownlint-cli + prettier, pinned to CI's versions
#   - go:     actionlint + gitleaks (Go binaries, no PyPI/npm wheel) — without them
#             check.sh silently self-skips those gates here and the failure only
#             surfaces in CI
#   - python: `uv sync` to materialize the locked dev environment up front
#
# Hook stdout is injected into the agent's context at session start, so emit one
# short line per step and send everything verbose to $LOG — a past session burned
# ~38KB of context on raw apt-get output.
#
# Only runs in the remote (web) environment — local dev machines already have
# these and shouldn't be reprovisioned. Idempotent and non-interactive. Every step
# soft-fails with a warning: a partially provisioned container is still usable,
# and check.sh self-skips whatever is missing (CI still enforces it).
set -euo pipefail

# Web-only: skip entirely on local machines.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Tool pins are shared with .github/workflows/ci.yml so lint results match CI
# exactly; bump them in scripts/gate_tool_pins.sh, not here.
# shellcheck source=scripts/gate_tool_pins.sh
. scripts/gate_tool_pins.sh

LOG="/tmp/session-start.log"
: >"$LOG"

log() { echo "[session-start] $*"; }

# 1. System packages (PortAudio + ffmpeg + shellcheck). apt-get is idempotent;
#    --no-install-recommends keeps the layer small. Skip the install if all three
#    are already present so resumed/cached containers don't re-run apt.
need_apt=0
command -v ffmpeg    >/dev/null 2>&1 || need_apt=1
command -v shellcheck >/dev/null 2>&1 || need_apt=1
ldconfig -p 2>/dev/null | grep -q portaudio || need_apt=1
if [ "$need_apt" = "1" ]; then
  export DEBIAN_FRONTEND=noninteractive
  # `apt-get update` can exit non-zero when unrelated third-party PPAs are
  # unreachable; the main Ubuntu archive (which has these packages) still
  # refreshes, so tolerate that failure and let the install be the real signal.
  apt-get update -qq >>"$LOG" 2>&1 || log "apt-get update reported errors (likely unrelated PPAs); continuing"
  if apt-get install -y --no-install-recommends libportaudio2 ffmpeg shellcheck >>"$LOG" 2>&1; then
    log "installed system deps (libportaudio2, ffmpeg, shellcheck)"
  else
    log "WARNING: apt-get install failed; some system deps may be missing (see $LOG)"
  fi
else
  log "system deps already present"
fi

# 2. Node lint CLIs — check.sh invokes `markdownlint` and `prettier` as global
#    binaries, not through uv, so they must be on PATH at CI's pinned versions
#    (the base image may ship a different prettier, and formatting output can
#    differ across versions).
if command -v markdownlint >/dev/null 2>&1; then
  log "markdownlint already present"
elif npm install -g "markdownlint-cli@${MARKDOWNLINT_VERSION}" >>"$LOG" 2>&1; then
  log "installed markdownlint-cli@${MARKDOWNLINT_VERSION}"
else
  log "WARNING: markdownlint install failed; check.sh's markdownlint gate will error (see $LOG)"
fi
if [ "$(prettier --version 2>/dev/null || true)" = "$PRETTIER_VERSION" ]; then
  log "prettier ${PRETTIER_VERSION} already present"
elif npm install -g "prettier@${PRETTIER_VERSION}" >>"$LOG" 2>&1; then
  log "installed prettier@${PRETTIER_VERSION}"
else
  log "WARNING: prettier install failed; check.sh self-skips its gate (CI still runs it; see $LOG)"
fi

# 3. Go gate binaries (actionlint + gitleaks), same pinned versions CI builds.
#    GOBIN=/usr/local/bin so they're on PATH for every later shell. `go install`
#    can't bake a comparable version string, so idempotency is presence-only —
#    fine, since containers are ephemeral and a pin bump rebuilds on cold start.
install_go_tool() { # $1 = binary name, $2 = module@version
  if command -v "$1" >/dev/null 2>&1; then
    log "$1 already present"
  elif GOBIN=/usr/local/bin go install "$2" >>"$LOG" 2>&1; then
    log "installed $1 ($2)"
  else
    log "WARNING: go install $2 failed; check.sh self-skips the $1 gate (CI still runs it; see $LOG)"
  fi
}
if command -v go >/dev/null 2>&1; then
  install_go_tool actionlint "$ACTIONLINT_MODULE"
  install_go_tool gitleaks "$GITLEAKS_MODULE"
else
  log "go not found; skipping actionlint/gitleaks (check.sh self-skips them; CI still runs them)"
fi

# 4. Git history — web containers start from a shallow clone, where origin/main
#    can exist with NO merge base to the session branch; check.sh's diff-scoped
#    tail gates (diff-cover/mutation) then crash with "fatal: ... no merge base"
#    instead of self-skipping, and the branch auto-update below can't merge.
#    Unshallow up front so both just work.
if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
  if git fetch --unshallow origin main >>"$LOG" 2>&1; then
    log "unshallowed clone (merge base with origin/main available for diff gates)"
  else
    log "WARNING: git fetch --unshallow failed; diff-cover/mutation gates may error (see $LOG)"
  fi
else
  log "clone already has full history"
fi

# 5. Keep the session branch current. Resumed web containers hold a clone frozen
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
        git merge --ff-only "origin/$branch" >>"$LOG" 2>&1
        log "fast-forwarded $branch to its remote tip"
      fi
    fi
    behind=$(git rev-list --count "HEAD..origin/main" 2>/dev/null || echo 0)
    if [ "$behind" -gt 0 ]; then
      if git merge --no-edit origin/main >>"$LOG" 2>&1; then
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

# 6. Python environment — materialize the locked dev env so the first `uv run`
#    doesn't pay the full sync cost mid-task. `uv` syncs the default dev group.
if uv sync >>"$LOG" 2>&1; then
  log "uv environment synced (locked dev group)"
else
  log "WARNING: uv sync failed (see $LOG)"
fi

log "provisioning complete (full output: $LOG)"
