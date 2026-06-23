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

echo "==> validate-pyproject (pyproject.toml schema)"
# Validate pyproject's standardized tables ([build-system]/[project]) against the PyPA
# JSON schemas. Run via uvx (like twine/codespell below) so it needs no dev-dep/uv.lock
# entry; --with packaging enables full requirement/license-expression checks. Unknown
# [tool.*] tables (ruff/mypy/pyright/…) are intentionally left to those tools.
uvx --with "packaging>=24.2" validate-pyproject pyproject.toml

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

echo "==> codespell (spell-check code, comments, docs)"
# Kubernetes' verify-spelling, generalized. Config (skips + ignore-words) is in
# [tool.codespell] in pyproject.toml. Run via uvx (like twine below) so it needs no
# entry in uv.lock; pre-commit also runs it. uvx self-skips if offline/unavailable.
if command -v uvx >/dev/null 2>&1; then
  uvx codespell .
else
  echo "   uvx not found; skipping (pre-commit + CI run codespell)"
fi

echo "==> json validity (all tracked + staged *.json)"
# Parse every JSON file so a malformed dashboard / vercel.json / fixture fails here
# instead of silently downstream (a bad dashboard just won't import). Validity only —
# recorded tests/fixtures/api/*.json are snapshots and must not be reformatted.
uv run python scripts/json_lint.py

echo "==> prettier (init template JS/CSS)"
# CI's runner has prettier on PATH; locally it's skipped with a notice if not
# installed, matching how shellcheck/swiftlint self-skip above.
if command -v prettier >/dev/null 2>&1; then
  prettier --check "aai_cli/init/templates/**/*.{js,css}"
else
  echo "   prettier not found; skipping (CI runs it)"
fi

echo "==> shellcheck"
# Static-lint this gate script. CI's ubuntu runner ships shellcheck;
# locally it's skipped with a notice if not installed.
if command -v shellcheck >/dev/null 2>&1; then
  # -x + --source-path=. let it follow the hook's `. scripts/gate_tool_pins.sh`
  # (paths resolve from the repo root, where this script always runs).
  shellcheck -x --source-path=. scripts/check.sh scripts/docker_build_check.sh \
    scripts/cut_release.sh scripts/gate_tool_pins.sh \
    .claude/hooks/session-start.sh .claude/hooks/require-gate-before-commit.sh
else
  echo "   shellcheck not found; skipping (CI runs it)"
fi

echo "==> actionlint (GitHub Actions workflow lint)"
# Static-lint the CI workflows the same way shellcheck covers shell scripts: catches
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

echo "==> unused snapshot/fixture gate"
# xdist disables syrupy's own unused-snapshot detection, so a renamed/deleted test can
# leave an orphaned .ambr or recorded API fixture behind. This static check catches it.
uv run python scripts/unused_fixtures_gate.py

echo "==> docs consistency gate (env vars / exit codes / command refs)"
# curl's "every option is documented" presubmit, generalized: REFERENCE.md/README.md must
# not drift from the code — every env var and exit code is documented, every `assembly …`
# example names a real command.
uv run python scripts/docs_consistency_gate.py

echo "==> docstring coverage gate (public API ratchet)"
# interrogate can't parse this codebase's PEP 695 generics, so an ast-based ratchet stands
# in: public-API docstring coverage may not drop below the floor in scripts/.
uv run python scripts/docstring_coverage_gate.py

echo "==> brew audit (Homebrew formula)"
# Lint the formula we ship (Formula/assembly.rb) the way Homebrew's own CI does, so a
# formula regression fails here instead of on the release PR. brew is macOS/Linuxbrew
# only, so this self-skips where it isn't installed (CI's release path has it).
#
# Homebrew 6+ disabled `brew audit [path ...]` — a formula must be audited by NAME,
# which means it has to live in a tap. Copy ours into an ephemeral local tap, audit by
# name, then remove it (works on both macOS and Linuxbrew, old and new). The explicit
# status capture keeps the tap cleanup running under `set -e` even when the audit fails.
if command -v brew >/dev/null 2>&1; then
  audit_tap="$(brew --repository)/Library/Taps/local/homebrew-aaiaudit"
  mkdir -p "$audit_tap/Formula"
  cp Formula/assembly.rb "$audit_tap/Formula/"
  audit_status=0
  brew audit --strict --formula local/aaiaudit/assembly || audit_status=$?
  rm -rf "$audit_tap"
  if [ "$audit_status" -ne 0 ]; then
    exit "$audit_status"
  fi
else
  echo "   brew not found; skipping (Homebrew CI / release runner has it)"
fi

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key. Exclude install (real per-template dep install, slow + network).
# All are uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
# -n auto parallelizes across CPUs (pytest-xdist); pytest-cov combines per-worker
# data, and the per-test --cov-context=test contexts the mutation gate below relies
# on survive that combine. The suite is order-independent (pytest-randomly), so
# splitting it across workers is safe.
uv run pytest -q --strict-config --strict-markers -n auto -m "not e2e and not install" --cov=aai_cli --cov-branch --cov-context=test --cov-report=term-missing --cov-report=xml --cov-fail-under=90

echo "==> Textual TUI coverage (>=90% on the textual-importing modules)"
# The project-wide 90% gate above is an average, so a TUI module can rot while the rest
# of the suite carries it. The Textual TUIs (`assembly code` / `live`) are the most
# layout-fragile, regression-prone surface in the repo (see tests/AGENTS.md), so hold
# them to their own >=90% floor. The module set is *derived* — every aai_cli file that
# imports `textual` — so a new TUI module is picked up automatically with no list to
# hand-maintain. Reuses the .coverage data the pytest step just wrote (no re-run), and
# counts branches because that data was collected with --cov-branch.
tui_modules="$(git grep -lP '^\s*(from|import) textual' -- 'aai_cli/**/*.py' | paste -sd, -)"
if [[ -z "$tui_modules" ]]; then
  echo "   no textual-importing modules found (the derive pattern is stale?)"
  exit 1
fi
uv run coverage report --include="$tui_modules" --fail-under=90

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
# justified by changing this gate deliberately. All hatch classes are count-gated
# against the merge-base with origin/main so mechanical edits — and *moving* an
# existing hatch (refactors relocate code wholesale, which an added-line scan would
# false-positive on) — don't fail, but net-new uses do.
if git rev-parse --verify --quiet origin/main >/dev/null; then
  # Diff and count against the MERGE-BASE, not the origin/main tip (which the
  # mutation gate and diff-cover already do). With many concurrent branches,
  # main moves constantly: a tip-based baseline makes this gate fail on a branch
  # the moment an unrelated merge lowers the count (e.g. removes an `Any`), even
  # though the branch itself added nothing. The merge-base only moves when the
  # branch itself rebases.
  gate_base="$(git merge-base origin/main HEAD || echo origin/main)"

  # Count hatch hits with ONE matcher on both sides so baseline and working tree compare
  # apples-to-apples. Using `rg` for the working tree but `git grep -E` for the baseline
  # diverged on `\b`: ERE ignores it on macOS (matching nothing) while rg honors it, so a
  # pre-existing time.sleep() inflated the working count over the baseline and failed this
  # gate on macOS though it passed on Linux. `git grep -P` (PCRE) handles `\b` identically
  # on both platforms; `--untracked` counts newly-added (unstaged) files the way rg did.
  # Patterns must be PCRE-valid (escape literal parens, e.g. `cast\(`).
  hatch_base() { { git grep -hP "$1" "$gate_base" -- "${@:2}" || true; } | wc -l | tr -d '[:space:]'; }
  hatch_work() { { git grep --untracked -hP "$1" -- "${@:2}" || true; } | wc -l | tr -d '[:space:]'; }

  hatch_pattern='# type: ignore|# noqa|pragma: no cover'
  base_hatch_count="$(hatch_base "$hatch_pattern" aai_cli tests)"
  work_hatch_count="$(hatch_work "$hatch_pattern" aai_cli tests)"
  if (( work_hatch_count > base_hatch_count )); then
    { git grep --untracked -nP "$hatch_pattern" -- aai_cli tests || true; } | tail -n 20
    echo "New static-analysis ignore/no-cover escape hatch found: ${work_hatch_count} current vs ${base_hatch_count} at the merge-base with origin/main. Refactor it or update the gate explicitly."
    exit 1
  fi

  # Test-suite escape hatches, same net-new-only policy: a skip/xfail is how an agent
  # makes a red test go away instead of fixing it, and time.sleep() is the classic
  # source of flakiness (use events/polling). The legitimate existing skips guard the
  # env-gated marker suites (e2e/install) and are counted at the merge-base, so they
  # don't trip this — and neither does moving one in a refactor; a genuinely-needed
  # new one must update this gate deliberately. Scoped to tests/ — production sleeps
  # are fine.
  shortcut_pattern='pytest\.skip\(|pytest\.xfail\(|@pytest\.mark\.(skip|xfail)|\btime\.sleep\('
  base_shortcut_count="$(hatch_base "$shortcut_pattern" tests)"
  work_shortcut_count="$(hatch_work "$shortcut_pattern" tests)"
  if (( work_shortcut_count > base_shortcut_count )); then
    { git grep --untracked -nP "$shortcut_pattern" -- tests || true; } | tail -n 20
    echo "New test skip/xfail/time.sleep found: ${work_shortcut_count} current vs ${base_shortcut_count} at the merge-base with origin/main. Fix the test (or sync properly) or update the gate explicitly."
    exit 1
  fi

  base_any_count="$(hatch_base "Any" aai_cli tests)"
  work_any_count="$(hatch_work "Any" aai_cli tests)"
  if (( work_any_count > base_any_count )); then
    echo "New Any usage found: ${work_any_count} current vs ${base_any_count} at the merge-base with origin/main."
    exit 1
  fi

  # `\b` so a domain word ending in "cast(" — `_forecast(`, `broadcast(` — isn't miscounted
  # as a typing.cast() escape hatch (`git grep -P` honors `\b` identically on macOS/Linux).
  base_cast_count="$(hatch_base '\bcast\(' aai_cli tests)"
  work_cast_count="$(hatch_work '\bcast\(' aai_cli tests)"
  if (( work_cast_count > base_cast_count )); then
    echo "New cast() usage found: ${work_cast_count} current vs ${base_cast_count} at the merge-base with origin/main."
    exit 1
  fi
else
  echo "   origin/main not found; skipping escape-hatch diff gate (CI provides it)"
fi

# CodeQL is NOT run here. It's the single slowest gate (~minutes) and is enforced in CI
# by codeql.yml, which runs the same security + quality suites on its own schedule and
# uploads them to GitHub's code-scanning/quality tabs. ci.yml's check job never ran it
# either (the hosted runner has no codeql on PATH, so this step self-skipped there), so
# dropping it from the local gate loses no CI coverage — it just keeps `check.sh` fast.
# To reproduce a code-scanning alert locally: `uv run python scripts/codeql_gate.py`.

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

# Record that this exact working tree passed, so the pre-commit gate hook
# (.claude/hooks/require-gate-before-commit.sh) can let a `git commit` through.
# Any later edit changes the signature and re-requires a green gate. Never fail
# the gate over the marker itself.
python3 scripts/gate_marker.py record || true

echo "All checks passed."
