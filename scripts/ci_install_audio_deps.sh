#!/usr/bin/env bash
# CI-only helper: install the Linux system audio deps the suite needs — libportaudio2
# (sounddevice's PortAudio backend) and ffmpeg (decodes non-WAV/URL audio for the
# `--sample` stream tests). Three Ubuntu jobs in .github/workflows/ci.yml install the
# identical pair, so the slow-mirror resilience below lives in one place.
#
# A plain `apt-get install ffmpeg` pulls a ~100-package / ~94 MB dependency closure, and
# against a degraded Azure Ubuntu mirror that download has repeatedly overrun the job's
# timeout — the step hung mid-download and was cancelled before the tests ran, bouncing
# the PR out of the merge queue. Two mitigations, mirroring the Windows ffmpeg step's
# bounded-retry philosophy:
#   * --no-install-recommends drops the optional VA-API/VDPAU/SDL/pocketsphinx recommends
#     that balloon the download; the ffmpeg binary and the libs the tests load remain.
#   * each apt call is wrapped in `timeout` (run under sudo so the killer is root and can
#     actually reap apt) and retried, so a connection that stalls on a bad mirror edge is
#     killed and retried instead of wedging the whole job; apt's own Acquire::Retries
#     handles transient per-file failures within an attempt.
set -euo pipefail

# Run one bounded apt-get attempt, retrying a stalled/failed call a few times.
apt_retry() {
  local attempt
  for attempt in 1 2 3; do
    if sudo timeout --kill-after=10s 120s apt-get "$@"; then
      return 0
    fi
    echo "::warning::apt-get $* failed or stalled (attempt ${attempt}/3); retrying" >&2
    sleep "$((attempt * 5))"
  done
  echo "::error::apt-get $* failed after 3 attempts" >&2
  return 1
}

apt_retry update -o Acquire::Retries=3
apt_retry install -y --no-install-recommends -o Acquire::Retries=3 libportaudio2 ffmpeg
