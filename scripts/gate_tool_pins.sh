# shellcheck shell=bash
# Single source of truth for the gate's non-Python tool pins. Python tools are
# pinned in pyproject.toml/uv.lock; these have no PyPI distribution
# (markdownlint/prettier are npm packages, actionlint/gitleaks are Go binaries,
# codeql is a GitHub release bundle), so their versions live here instead.
# Sourced by both provisioning paths:
#   - .github/workflows/ci.yml (the CI runner)
#   - .claude/hooks/session-start.sh (Claude Code on the web containers)
# Bump a pin here and both environments pick it up together.
export MARKDOWNLINT_VERSION="0.45.0"
export PRETTIER_VERSION="3.8.3"
export ACTIONLINT_MODULE="github.com/rhysd/actionlint/cmd/actionlint@v1.7.7"
export GITLEAKS_MODULE="github.com/zricethezav/gitleaks/v8@v8.21.2"
# The CLI+query-pack bundle check.sh's codeql gate runs (codeql.yml's CI runs use
# the version pinned to the codeql-action release instead; keep them roughly in step).
export CODEQL_BUNDLE_VERSION="codeql-bundle-v2.25.6"
