---
name: plan-feature
description: Plan a new assembly CLI feature the way this repo does — drive the superpowers planning skill, then persist the spec + plan into docs/superpowers/ under this repo's naming and verification conventions. Use before building anything non-trivial.
disable-model-invocation: true
argument-hint: "<short feature description>"
---

# plan-feature

This repo plans features with the **superpowers** skill (the existing plans say
so: `REQUIRED SUB-SKILL: superpowers:subagent-driven-development` /
`superpowers:executing-plans`). This skill is a thin wrapper: it runs the
superpowers planning flow, then drops the artifacts into `docs/superpowers/`
under this repo's filenames and verification gates. It does **not** reinvent the
planning methodology, and it does **not** start writing feature code — the
output is two reviewable documents.

## 1. Drive the superpowers planning flow

Use the superpowers skill to do the actual work — brainstorming the design, then
writing the task-by-task plan. (If the superpowers plugin isn't installed in
this environment, say so and follow the structure of the existing files in
`docs/superpowers/specs/` and `plans/` by hand.)

Ground the design in this repo's real architecture — read `CLAUDE.md` first:
Typer sub-apps under `aai_cli/commands/` run through `context.run_command`; the
API key never touches disk or argv; errors are `CLIError`s on stderr (data on
stdout); new SDK calls follow `client.py`'s auth-failure/`APIError` shape. The
design must call out which of those guarantees it touches.

## 2. Persist with this repo's names

Superpowers' outputs land here, matching the existing files exactly:

- Spec: `docs/superpowers/specs/<DATE>-<SLUG>-design.md`
- Plan: `docs/superpowers/plans/<DATE>-<SLUG>.md`

where `DATE` is today (`YYYY-MM-DD`, the repo's `currentDate` — not a guess) and
`SLUG` is short kebab-case (e.g. `show-code`, `cli-color-theme`). If a file
already exists at either path, stop and ask whether to amend rather than
overwrite.

## 3. Bake in this repo's verification gates

Whatever plan superpowers produces, its self-review / done-criteria must include
this repo's gates (the implementation isn't done until these pass):

- [ ] `./scripts/check.sh` is green (ruff, mypy, pyright, markdownlint, shellcheck, pytest with the **90% branch-coverage gate**, build, twine).
- [ ] `/review-changes` run — it runs the `code-review` skill on the diff, plus (for security-sensitive diffs touching `aai_cli/auth/`, `config.py`, `environments.py`, `client.py`, or any `subprocess` call) the `security-review` skill followed by the project's `security-reviewer` agent, and the `template-contract-reviewer` agent (if it touches `aai_cli/init/templates/` or wheel packaging).
- [ ] Security guarantees intact: key never on disk/argv; env↔credential binding preserved.

## 4. Hand off

Report the two file paths created and a one-line summary of each. Tell the user
the design is ready for review and that implementation proceeds task-by-task via
superpowers, running `/check` before pushing and `/review-changes` on the diff.
Do not begin implementing in this skill.
