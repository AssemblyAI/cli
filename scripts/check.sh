#!/usr/bin/env bash
# Lint, typecheck, and test. Run locally before pushing; CI runs this on every PR.
set -euo pipefail

cd "$(dirname "$0")/.."

# Run the Python tools through `uv run` so they use the project's locked
# environment (pyproject + uv.lock), not whatever happens to be on PATH. This keeps
# results reproducible and consistent with `uv run` used everywhere else.
#
# `--extra dev` installs the optional-dependencies group (pytest, hypothesis,
# fastapi, numpy, …) into the uv environment. `uv run` does NOT install extras by
# default, so without this the type-checkers and pytest can't resolve those imports
# (mypy hides it via ignore_missing_imports; pyright reports reportMissingImports).
UV_RUN=(uv run --extra dev)

echo "==> ruff check (src + tests)"
"${UV_RUN[@]}" ruff check .

echo "==> ruff format --check (src + tests)"
"${UV_RUN[@]}" ruff format --check .

echo "==> mypy (src + tests)"
"${UV_RUN[@]}" mypy  # files = ["aai_cli", "tests"] in pyproject.toml

echo "==> pyright (src + tests)"
"${UV_RUN[@]}" pyright  # include = ["aai_cli", "tests"] in [tool.pyright]

echo "==> markdownlint (docs/ is generated, so excluded)"
markdownlint "**/*.md" --ignore docs --ignore node_modules --ignore .pytest_cache

echo "==> pytest (with branch-coverage gate)"
# Exclude e2e: they drive the CLI as a subprocess (uncounted by coverage) and need
# a live API key + kokoro. And exclude install (real per-template dep install,
# slow + network), also uncounted by coverage. Run them with:
#   uv run pytest -m e2e
#   uv run pytest -m install
"${UV_RUN[@]}" pytest -q -m "not e2e and not install" --cov=aai_cli --cov-branch --cov-report=term-missing --cov-fail-under=90

echo "==> build + twine check (PyPI publish readiness)"
# Build sdist + wheel into ./dist, then validate the metadata and README render
# the way PyPI requires. --strict fails on any warning (e.g. a missing readme).
rm -rf dist
uv build
uvx twine check --strict dist/*

echo "All checks passed."
