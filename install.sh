#!/bin/sh
# Install the AssemblyAI CLI (`aai`) without cloning the repo:
#
#   curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh
#
# Overridable via env: AAI_REPO (owner/name), AAI_REF (branch/tag/sha).
set -eu

REPO="${AAI_REPO:-AssemblyAI/cli}"
REF="${AAI_REF:-main}"
SPEC="git+https://github.com/${REPO}.git@${REF}"

info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
err() { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; }

# --- Require Python 3.10+ -------------------------------------------------
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  err "Python 3.10+ is required, but no python3 was found on PATH."
  exit 1
fi
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  err "Python 3.10+ is required (found $("$PY" -V 2>&1))."
  exit 1
fi

# --- Install (prefer pipx for an isolated env; fall back to pip --user) ----
if command -v pipx >/dev/null 2>&1; then
  info "Installing aai with pipx from ${REPO}@${REF}..."
  pipx install --force "$SPEC"
else
  info "pipx not found; installing with pip --user from ${REPO}@${REF}..."
  "$PY" -m pip install --user --upgrade "$SPEC"
fi

# --- Next steps -----------------------------------------------------------
if command -v aai >/dev/null 2>&1; then
  info "Installed. Next: run 'aai login', then 'aai transcribe --sample'."
else
  info "Installed, but 'aai' isn't on your PATH yet."
  info "Run 'pipx ensurepath' (or add ~/.local/bin to PATH), then restart your shell."
fi
