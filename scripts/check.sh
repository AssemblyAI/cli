#!/usr/bin/env bash
# Lint, typecheck, and test. Run locally before pushing; CI runs this on every PR.
set -euo pipefail

cd "$(dirname "$0")/.."

# Run the Python tools through `uv run` so they use the project's locked
# environment (pyproject + uv.lock), not whatever happens to be on PATH. This keeps
# results reproducible and consistent with `uv run` used everywhere else.
#
# The dev dependencies live in [dependency-groups].dev, which uv installs by
# default (see [tool.uv] default-groups), so `uv run` already has pytest,
# hypothesis, fastapi, etc. — no `--extra`/`--group` flag needed here.

cleanup_generated_code_dir() {
  if [[ -n "${generated_code_dir:-}" ]]; then
    rm -rf "$generated_code_dir"
  fi
}

echo "==> uv lock freshness"
uv lock --check

echo "==> ruff check (src + tests)"
uv run ruff check .

echo "==> ruff format --check (src + tests)"
uv run ruff format --check .

echo "==> mypy (src + tests)"
uv run mypy  # files = ["aai_cli", "tests"] in pyproject.toml

echo "==> pyright (src strict)"
uv run pyright  # include = ["aai_cli"] in [tool.pyright]

echo "==> pyright (tests standard)"
uv run pyright -p pyrightconfig.tests.json

echo "==> vulture (dead-code gate, src + tests)"
uv run vulture

echo "==> deptry (dependency hygiene)"
uv run deptry .

echo "==> import-linter (architecture contracts)"
uv run lint-imports

echo "==> max file length (500-line gate, src + tests + scripts)"
# Keep modules small enough for humans and AI coding agents to hold in context.
# Raising the cap is a deliberate edit to scripts/max_file_length.py, not a per-file
# exception.
uv run python scripts/max_file_length.py

echo "==> xenon (cyclomatic complexity gate, src only)"
# Fail the build if any function gets too branchy. Grades map to cyclomatic
# complexity: A=1-5, B=6-10, C=11-20, ... Thresholds:
#   --max-absolute B : no single function may exceed CC 10 (grade B). Pairs with ruff's
#                      mccabe max-complexity=10 (C901); xenon/radon also counts boolean
#                      operators, so it's the stricter of the two on the same number.
#                      Raw length/arg limits live in ruff (PLR0915/C901/PLR0913) —
#                      xenon only measures branching.
#   --max-modules  A : no file's *average* may exceed grade A (CC <= 5), so no single
#                      module is allowed to become a complexity hotspot on average.
#   --max-average  A : the project-wide average must stay grade A (CC <= 5).
# Tests are excluded (not shipped); only the aai_cli package is gated.
uv run xenon --max-absolute B --max-modules A --max-average A aai_cli

echo "==> swiftlint (macOS audio helper)"
if command -v swiftlint >/dev/null 2>&1; then
  swiftlint lint --no-cache --strict aai_cli/streaming/macos_system_audio.swift
else
  echo "   swiftlint not found; skipping (install with: brew install swiftlint)"
fi

echo "==> swift compile (macOS audio helper)"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "   not macOS; skipping compile for macOS-only frameworks"
elif command -v swiftc >/dev/null 2>&1; then
  swift_module_cache="$(mktemp -d)"
  swift_helper="$swift_module_cache/aai-macos-audio-check"
  swift_error="$swift_module_cache/aai-macos-audio-check.err"
  swiftc -parse-as-library aai_cli/streaming/macos_system_audio.swift \
    -module-cache-path "$swift_module_cache" \
    -O \
    -framework ScreenCaptureKit \
    -framework AVFoundation \
    -framework CoreMedia \
    -framework CoreGraphics \
    -o "$swift_helper"
  if "$swift_helper" --unknown-check-flag 2>"$swift_error"; then
    echo "   expected Swift helper argument validation to fail"
    exit 1
  fi
  if ! grep -q "Unknown argument: --unknown-check-flag" "$swift_error"; then
    cat "$swift_error"
    exit 1
  fi
  rm -rf "$swift_module_cache"
else
  echo "   swiftc not found; skipping (macOS system audio builds on first use)"
fi

echo "==> markdownlint (docs/ is generated, so excluded)"
markdownlint "**/*.md" --ignore docs --ignore node_modules --ignore .pytest_cache

echo "==> prettier (init template JS/CSS)"
# CI's runner has prettier on PATH; locally it's skipped with a notice if not
# installed, matching how shellcheck/swiftlint self-skip above.
if command -v prettier >/dev/null 2>&1; then
  prettier --check "aai_cli/init/templates/**/*.{js,css}"
else
  echo "   prettier not found; skipping (CI runs it)"
fi

echo "==> shellcheck (install.sh)"
# Static-lint the public install script and this gate script. CI's ubuntu runner ships shellcheck;
# locally it's skipped with a notice if not installed.
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck install.sh scripts/check.sh
else
  echo "   shellcheck not found; skipping (CI runs it)"
fi

echo "==> actionlint (GitHub Actions workflow lint)"
# Static-lint the CI workflows the same way shellcheck covers install.sh: catches
# bad expressions, undefined needs/matrix refs, and shell bugs inside `run:` blocks.
# Go binary (no PyPI wheel), so it self-skips locally and CI installs it (see ci.yml).
if command -v actionlint >/dev/null 2>&1; then
  actionlint
else
  echo "   actionlint not found; skipping (CI runs it)"
fi

echo "==> zizmor (GitHub Actions security audit)"
# Audits the workflows for CI security issues (script injection via untrusted
# ${{ github.* }} interpolation, over-broad token permissions, unpinned actions).
# Pip-installable, so it runs in the locked env as a hard gate like ruff/mypy.
# --offline keeps it deterministic (skips audits that would query the GitHub API).
uv run zizmor --offline .github/workflows

echo "==> gitleaks (secret scan)"
# Defends the project's core promise that credentials never land in the repo (the API
# key lives only in the OS keyring). Scans the working tree; obviously-fake test/doc
# fixtures are allowlisted in .gitleaks.toml. Go binary, so it self-skips locally and
# CI installs it (see ci.yml).
if command -v gitleaks >/dev/null 2>&1; then
  gitleaks dir --no-banner --redact -c .gitleaks.toml .
else
  echo "   gitleaks not found; skipping (CI runs it)"
fi

echo "==> generated --show-code compile gate"
generated_code_dir="$(mktemp -d)"
trap cleanup_generated_code_dir EXIT
uv run python scripts/generated_code_compile_gate.py "$generated_code_dir"
uv run python -m compileall -q "$generated_code_dir"
cleanup_generated_code_dir
trap - EXIT

echo "==> init template contract/import gate"
uv run python scripts/template_contract_gate.py

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key. Exclude install (real per-template dep install, slow + network)
# and install_script (builds a wheel and runs install.sh for real; slow, needs
# network + uv/pipx). All are uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
#   uv run pytest -m install_script
uv run pytest -q --strict-config --strict-markers -m "not e2e and not install and not install_script" --cov=aai_cli --cov-branch --cov-context=test --cov-report=term-missing --cov-report=xml --cov-fail-under=90

echo "==> diff-cover (patch coverage: every changed line must be tested)"
# The 90% gate above is project-wide, so new code can ride on the existing suite and
# stay untested. diff-cover requires 100% coverage of the lines changed versus the
# merge-base with origin/main (uses coverage.xml from the pytest step). Genuinely
# unreachable defensive lines can be marked `# pragma: no cover`. Skipped with a
# notice when origin/main isn't present (e.g. a shallow clone of just the branch).
if git rev-parse --verify --quiet origin/main >/dev/null; then
  uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100
else
  echo "   origin/main not found; skipping patch-coverage gate (CI provides it)"
fi

echo "==> mutation gate (diff-scoped: a changed line's test must fail when it breaks)"
# Coverage proves a changed line ran; this proves a test would FAIL if it broke.
# Mutates only the lines changed vs origin/main and reruns just the tests that cover
# each mutant (per-test contexts from the .coverage written above). Survivors mean a
# weak/missing assertion — fix it or mark the line `# pragma: no mutate`. Self-skips
# when origin/main is absent (same as diff-cover).
if git rev-parse --verify --quiet origin/main >/dev/null; then
  uv run python scripts/mutation_gate.py origin/main
else
  echo "   origin/main not found; skipping mutation gate (CI provides it)"
fi

echo "==> no new static-analysis escape hatches"
# Existing escape hatches are tolerated for now; new ones must be refactored away or
# justified by changing this gate deliberately. Broad noqa/type-ignore/no-cover are
# checked by added diff lines. `Any` and `cast(` are count-gated against origin/main
# so mechanical edits to existing uses don't fail, but net-new uses do.
if git rev-parse --verify --quiet origin/main >/dev/null; then
  escape_hatches="$(git diff -U0 origin/main -- aai_cli tests \
    | rg '^\+.*(# type: ignore|# noqa|pragma: no cover)' || true)"
  if [[ -n "$escape_hatches" ]]; then
    printf '%s\n' "$escape_hatches"
    echo "New static-analysis ignore/no-cover escape hatch found; refactor it or update the gate explicitly."
    exit 1
  fi

  # Test-suite escape hatches, same net-new-only policy: a skip/xfail is how an agent
  # makes a red test go away instead of fixing it, and time.sleep() is the classic
  # source of flakiness (use events/polling). The legitimate existing skips guard the
  # env-gated marker suites (e2e/install/install_script) and live on origin/main, so
  # they aren't added diff lines and don't trip this; a genuinely-needed new one must
  # update this gate deliberately. Scoped to tests/ — production sleeps are fine.
  test_shortcuts="$(git diff -U0 origin/main -- tests \
    | rg '^\+.*(pytest\.skip\(|pytest\.xfail\(|@pytest\.mark\.(skip|xfail)|\btime\.sleep\()' || true)"
  if [[ -n "$test_shortcuts" ]]; then
    printf '%s\n' "$test_shortcuts"
    echo "New test skip/xfail/time.sleep found; fix the test (or sync properly) or update the gate explicitly."
    exit 1
  fi

  base_any_count="$({ git grep -n "Any" origin/main -- aai_cli tests || true; } | wc -l | tr -d '[:space:]')"
  work_any_count="$({ rg -n "Any" aai_cli tests || true; } | wc -l | tr -d '[:space:]')"
  if (( work_any_count > base_any_count )); then
    echo "New Any usage found: ${work_any_count} current vs ${base_any_count} on origin/main."
    exit 1
  fi

  base_cast_count="$({ git grep -n "cast(" origin/main -- aai_cli tests || true; } | wc -l | tr -d '[:space:]')"
  work_cast_count="$({ rg -n "cast\\(" aai_cli tests || true; } | wc -l | tr -d '[:space:]')"
  if (( work_cast_count > base_cast_count )); then
    echo "New cast() usage found: ${work_cast_count} current vs ${base_cast_count} on origin/main."
    exit 1
  fi
else
  echo "   origin/main not found; skipping escape-hatch diff gate (CI provides it)"
fi

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

echo "All checks passed."
