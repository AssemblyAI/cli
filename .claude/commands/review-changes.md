---
description: Review the current changes with the code-review skill plus the aai CLI's specialized reviewers (security + template contract), scoped to what actually changed.
argument-hint: "[git ref to diff against, default: HEAD]"
allowed-tools: Bash(git diff *), Bash(git status *), Bash(git log *), Task, Skill
---

Review the current working changes using the general `code-review` skill and this project's specialized subagents. Be surgical: the code review always runs on the diff; only run a specialized reviewer if the diff actually touches its area.

## 1. Scope the diff

Run `git status --short` and `git diff --stat ${1:-HEAD}` (and `git diff ${1:-HEAD}` for detail) to see what changed.

## 2. Run the general code review (always)

Invoke the **`code-review`** skill on the current changes for correctness bugs and reuse/simplification/efficiency cleanups. This runs on every `/review-changes`, regardless of which files changed.

## 3. Dispatch the relevant specialized reviewers (in parallel)

- If the diff touches **`aai_cli/auth/`, `config.py`, `environments.py`, `client.py`, or any `subprocess` call** → run the **`security-review`** skill for the general security pass, **then** dispatch the project's **`security-reviewer`** agent for the AssemblyAI-specific guarantees the generic skill won't know (key never on disk/argv, env↔credential binding, `public-token-*` only, fixed-argv subprocess shell-outs).
- If the diff touches **`aai_cli/init/templates/`, `aai_cli/init/scaffold.py`, `aai_cli/init/templates.py`, or the wheel-packaging config in `pyproject.toml`** → dispatch the **`template-contract-reviewer`** agent on those changes.
- If the diff touches none of those sensitive areas, say so and skip the specialized reviewers — don't manufacture a review. (The `code-review` skill from step 2 still ran.)

Pass each agent the exact list of changed files in its scope so it reviews the diff, not the whole repo.

## 4. Synthesize

Combine the `code-review` findings and any specialized-reviewer findings into one ranked report (severity → file:line → fix). Call out anything that would break the CLI's security guarantees (key never on disk/argv, env↔credential binding) or the template packaging contract. If everything comes back clean, state that plainly.
