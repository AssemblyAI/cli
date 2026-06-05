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

echo "==> ruff check (src + tests)"
uv run ruff check .

echo "==> ruff format --check (src + tests)"
uv run ruff format --check .

echo "==> mypy (src + tests)"
uv run mypy  # files = ["aai_cli", "tests"] in pyproject.toml

echo "==> pyright (src + tests)"
uv run pyright  # include = ["aai_cli", "tests"] in [tool.pyright]

echo "==> xenon (cyclomatic complexity gate, src only)"
# Fail the build if any function gets too branchy. Grades map to cyclomatic
# complexity: A=1-5, B=6-10, C=11-20, ... Thresholds:
#   --max-absolute B : no single function may exceed CC 10 (grade B).
#   --max-modules  B : no file's average may exceed grade B.
#   --max-average  A : the project-wide average must stay grade A (CC <= 5).
# Tests are excluded (not shipped); only the aai_cli package is gated.
uv run xenon --max-absolute B --max-modules B --max-average A aai_cli

echo "==> markdownlint (docs/ is generated, so excluded)"
markdownlint "**/*.md" --ignore docs --ignore node_modules --ignore .pytest_cache

echo "==> shellcheck (install.sh)"
# Static-lint the public install script. CI's ubuntu runner ships shellcheck;
# locally it's skipped with a notice if not installed.
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck install.sh
else
  echo "   shellcheck not found; skipping (CI runs it)"
fi

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key. Exclude install (real per-template dep install, slow + network)
# and install_script (builds a wheel and runs install.sh for real; slow, needs
# network + uv/pipx). All are uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
#   uv run pytest -m install_script
uv run pytest -q -m "not e2e and not install and not install_script" --cov=aai_cli --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=90

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

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

echo "All checks passed."
