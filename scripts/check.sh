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

echo "==> markdownlint (docs/ is generated, so excluded)"
markdownlint "**/*.md" --ignore docs --ignore node_modules --ignore .pytest_cache

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key + kokoro. And exclude install (real per-template dep install,
# slow + network), also uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
uv run pytest -q -m "not e2e and not install" --cov=aai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

echo "All checks passed."
