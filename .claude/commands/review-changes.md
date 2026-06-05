---
description: Review the current changes with the aai CLI's specialized reviewers (security + template contract), scoped to what actually changed.
argument-hint: "[git ref to diff against, default: HEAD]"
allowed-tools: Bash(git diff *), Bash(git status *), Bash(git log *), Task
---

Review the current working changes using this project's specialized subagents. Be surgical: only run a reviewer if the diff actually touches its area.

## 1. Scope the diff

Run `git status --short` and `git diff --stat ${1:-HEAD}` (and `git diff ${1:-HEAD}` for detail) to see what changed.

## 2. Dispatch the relevant reviewers (in parallel)

- If the diff touches **`aai_cli/auth/`, `config.py`, `environments.py`, `client.py`, or any `subprocess` call** → dispatch the **`security-reviewer`** agent on those changes.
- If the diff touches **`aai_cli/init/templates/`, `aai_cli/init/scaffold.py`, `aai_cli/init/templates.py`, or the wheel-packaging config in `pyproject.toml`** → dispatch the **`template-contract-reviewer`** agent on those changes.
- If the diff touches neither sensitive area, say so and skip — don't manufacture a review.

Pass each agent the exact list of changed files in its scope so it reviews the diff, not the whole repo.

## 3. Synthesize

Combine the findings into one ranked report (severity → file:line → fix). Call out anything that would break the CLI's security guarantees (key never on disk/argv, env↔credential binding) or the template packaging contract. If both reviewers come back clean, state that plainly.
