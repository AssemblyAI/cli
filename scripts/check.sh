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

echo "==> pytest (with coverage gate)"
pytest -q --cov=assemblyai_cli --cov-report=term-missing --cov-fail-under=90

echo "All checks passed."
