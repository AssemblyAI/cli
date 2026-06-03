#!/usr/bin/env bash
# Lint, typecheck, and test. Run locally before pushing; CI runs this on every PR.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> ruff check"
ruff check .

echo "==> ruff format --check"
ruff format --check .

echo "==> mypy"
mypy

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key + kokoro. Run them with: pytest -m e2e
pytest -q -m "not e2e" --cov=assemblyai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90

echo "All checks passed."
