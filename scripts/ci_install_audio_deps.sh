#!/usr/bin/env bash
# CI-only helper: install the Linux system audio deps the suite needs — libportaudio2
# (sounddevice's PortAudio backend) and ffmpeg (decodes non-WAV/URL audio for the
# `--sample` stream tests; the require_ffmpeg probe needs it on PATH). Three Ubuntu jobs
# in .github/workflows/ci.yml need the identical pair, so the slow-mirror resilience
# below lives in one place.
#
# The Azure Ubuntu apt mirror periodically degrades to a crawl (~100 KB/s). A plain
# `apt-get install ffmpeg` pulls ffmpeg's ~62 MB codec dependency closure, which then
# overruns the job timeout mid-download and gets cancelled before the tests run, bouncing
# the PR out of the merge queue. So, mirroring the Windows ffmpeg step's strategy of
# falling back to a static build off GitHub's release CDN:
#   * libportaudio2 has no portable prebuilt, so it always comes from apt — but it's tiny,
#     so a bounded retry rides out a slow mirror.
#   * ffmpeg is fetched as a static build from GitHub's release CDN (BtbN/FFmpeg-Builds —
#     the same origin the Windows job uses), bypassing apt's heavy codec chain on the
#     flaky mirror entirely, and prepended to GITHUB_PATH so later steps see it.
set -euo pipefail

# Run one bounded apt-get attempt, retrying a stalled/failed call a couple of times. A
# stalled mirror connection is killed by `timeout` (run under sudo so the killer is root
# and can reap apt) rather than wedging the whole job; the 3rd failure falls through.
apt_retry() {
  local attempt
  for attempt in 1 2 3; do
    if sudo timeout --kill-after=10s 120s apt-get "$@"; then
      return 0
    fi
    if [ "$attempt" -lt 3 ]; then
      echo "::warning::apt-get $* failed or stalled (attempt ${attempt}/3); retrying" >&2
      sleep "$((attempt * 5))"
    fi
  done
  echo "::warning::apt-get $* failed after 3 attempts" >&2
  return 1
}

# A stale list is fine (the runner image's apt cache is recent), so don't let a slow
# `update` be fatal; libportaudio2 itself has no CDN fallback, so that one must succeed.
apt_retry update -o Acquire::Retries=3 || true
apt_retry install -y --no-install-recommends -o Acquire::Retries=3 libportaudio2

# ffmpeg from GitHub's release CDN, not apt: a static, self-contained build off a reliable
# origin sidesteps the 62 MB codec download the degraded apt mirror kept failing to serve.
url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
dest="${RUNNER_TEMP:-/tmp}/ffmpeg-static"
mkdir -p "$dest"
curl -fsSL --retry 3 --retry-all-errors "$url" | tar -xJ -C "$dest" --strip-components=1
echo "$dest/bin" >> "${GITHUB_PATH:-/dev/null}"
export PATH="$dest/bin:$PATH"

command -v ffmpeg >/dev/null || {
  echo "::error::ffmpeg unavailable after setup" >&2
  exit 1
}
ffmpeg -version >/dev/null
echo "audio deps ready: libportaudio2 + ffmpeg ($(command -v ffmpeg))"
