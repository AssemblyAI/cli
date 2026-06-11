# shellcheck shell=bash
# Single source of truth for the gate's non-Python tool pins. Python tools are
# pinned in pyproject.toml/uv.lock; these four have no PyPI distribution
# (markdownlint/prettier are npm packages, actionlint/gitleaks are Go binaries),
# so their versions live here instead. Sourced by both provisioning paths:
#   - .github/workflows/ci.yml (the CI runner)
#   - .claude/hooks/session-start.sh (Claude Code on the web containers)
# Bump a pin here and both environments pick it up together.
export MARKDOWNLINT_VERSION="0.45.0"
export PRETTIER_VERSION="3.8.3"
export ACTIONLINT_MODULE="github.com/rhysd/actionlint/cmd/actionlint@v1.7.7"
export GITLEAKS_MODULE="github.com/zricethezav/gitleaks/v8@v8.21.2"
