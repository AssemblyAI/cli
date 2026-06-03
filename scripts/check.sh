#!/usr/bin/env bash
# Lint, typecheck, and test. Run locally before pushing; CI runs this on every PR.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff check (src + tests)"
ruff check .

echo "==> ruff format --check (src + tests)"
ruff format --check .

echo "==> mypy (src + tests)"
mypy  # files = ["assemblyai_cli", "tests"] in pyproject.toml

echo "==> markdownlint (docs/ is generated, so excluded)"
markdownlint "**/*.md" --ignore docs --ignore node_modules --ignore .pytest_cache

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key + kokoro. Run them with: pytest -m e2e
pytest -q -m "not e2e" --cov=assemblyai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

echo "All checks passed."
