#!/usr/bin/env bash
# Build each `aai init` template's Docker image to prove the shipped Dockerfile works
# end to end: scaffold the template (which renames `dockerignore` -> `.dockerignore`
# and writes a placeholder `.env`), then `docker build` the result. This catches a
# broken Dockerfile, an uninstallable requirement, or a bad COPY layout — none of which
# the static template-contract gate can see.
#
# Run from the repo root:  ./scripts/docker_build_check.sh
#
# Self-skips when Docker is unavailable, the same way scripts/check.sh skips its
# optional linters when their tools are absent — so it's safe to call from CI
# unconditionally.
set -euo pipefail

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "==> docker build check: Docker not available; skipping"
  exit 0
fi

templates=(audio-transcription live-captions voice-agent)

workdir="$(mktemp -d)"
cleanup() {
  rm -rf "$workdir"
  for t in "${templates[@]}"; do
    docker image rm "aai-template-${t}:dockercheck" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

for t in "${templates[@]}"; do
  app="$workdir/$t"
  echo "==> scaffolding $t"
  uv run aai init "$t" "$app" --no-install >/dev/null
  echo "==> docker build $t"
  docker build --quiet -t "aai-template-${t}:dockercheck" "$app"
done

echo "All ${#templates[@]} template Docker images build."
