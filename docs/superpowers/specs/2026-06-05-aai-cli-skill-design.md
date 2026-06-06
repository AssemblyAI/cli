# aai-cli skill — design

## Problem

When an agent (Claude Code or otherwise) drives the installed `aai` CLI, it has
no grounding in the CLI's command surface or — more importantly — its
non-obvious operational rules: the auth/env credential binding, the
no-`--api-key`-on-run-commands rule, the stdout/stderr split, and the
`--json`-when-piped behavior. It guesses, and the guesses leak keys into argv,
mix sandbox credentials with production, or parse human-formatted output.

Vercel ships `skills/vercel-cli` for exactly this reason: a model-invocable
skill that teaches an agent how to use the `vercel` CLI safely. This design
adds the equivalent for `aai`.

This skill is distinct from the existing **`assemblyai`** skill (installed from
`AssemblyAI/assemblyai-skill` via `aai claude install`), which covers the
**APIs/SDKs**. The new skill covers the **`aai` CLI** specifically. They
coexist.

## Goals

- A model-invocable `aai-cli` skill that auto-triggers when an agent is using
  the `aai` CLI, teaching it the command surface and the operational gotchas.
- Mirror Vercel's layout: lean `SKILL.md` with a decision tree + anti-patterns,
  routing to `references/*.md`.
- Wire the skill into `aai claude install` / `status` / `remove`, alongside the
  existing docs MCP and `assemblyai` skill.

## Non-goals (YAGNI)

- No `command/` subdirectory (Vercel has one; we don't need it).
- No new CLI command and no changes to the CLI's behavior.
- No auto-generated reference content — content is hand-written and verified
  against `aai <cmd> --help`.

## Placement & structure

New directory at the repo root, mirroring `vercel/vercel`'s `skills/vercel-cli`:

```
skills/aai-cli/
  SKILL.md
  references/
    transcription.md   # transcribe, stream, agent, llm
    history.md         # transcripts, sessions
    account.md         # login, logout, whoami, balance, usage, limits, keys, audit
    setup.md           # init, samples, doctor, claude, version
```

The reference split **mirrors the CLI's own `aai --help` groups** (Quick
Start / Setup & Tools / Transcription & AI / History / Account), so the skill's
organization stays self-consistent with the tool it documents. `transcribe` and
`stream` carry the most flags and get the deepest treatment.

### `SKILL.md`

Frontmatter — **model-invocable** (no `disable-model-invocation`):

```yaml
---
name: aai-cli
description: <broad capability list so it triggers whenever an agent drives the
  aai CLI — transcription, real-time streaming, voice agents, LLM Gateway,
  transcript/session history, login/auth, account balance/usage/limits/keys/
  audit, project scaffolding (init/samples), doctor, and Claude Code setup>
---
```

Body sections, in order:

1. **Intro** — what `aai` is; "run `aai <cmd> --help`; it is the source of
   truth for flags."
2. **Critical: auth & environment** (highest-leverage section):
   - Auth: `aai login` (browser) **or** the `ASSEMBLYAI_API_KEY` env var. Key
     resolution order: env → OS keyring. **Run commands deliberately expose no
     `--api-key` flag** so keys can't leak into `ps`/shell history.
   - Environment binding: `--env` / `AAI_ENV` / `--sandbox`. **A credential is
     only valid against the environment that minted it** (a sandbox key fails
     against production and vice-versa). Default env is currently `sandbox000`.
   - Profiles: `--profile`.
3. **Output contract for agents** — data → stdout, errors → stderr; `--json` is
   **auto-enabled when piped or agent-run**, so parse stdout as JSON.
4. **Quick start** — a short copy-paste block (login → transcribe → init).
5. **Decision tree** — routes to the four `references/*.md` files.
6. **Anti-patterns**:
   - Passing `--api-key` to a run command — it doesn't exist; use `aai login`
     or `ASSEMBLYAI_API_KEY`.
   - Using a credential against the wrong environment (sandbox vs production).
   - Running a command before `aai login` / setting a key → auth failure; run
     `aai doctor` to diagnose setup.
   - Assuming `pip install assemblyai-cli` works — the PyPI name is squatted by
     a third party; use the project's official install path.
   - Forgetting that `--show-code` on `transcribe` / `stream` / `agent` emits a
     ready-to-run SDK script (no API call needed).

All command and flag examples MUST be verified against real `aai <cmd> --help`
output before the skill is considered done.

## Distribution wiring (`aai_cli/commands/claude.py`)

Add a third step to `install`, `status`, and `remove`, mirroring the existing
`_install_skill` / `_skill_status` / `_remove_skill` shape exactly:

- Install via `npx skills add ... --global` for **this** repo
  (`AssemblyAI/cli`), idempotent with a `--force` path, gracefully **skipped**
  (not failed) when `npx` is missing — identical to the existing skill step.
- Detect installation at `~/.claude/skills/aai-cli/SKILL.md`, honoring
  `CLAUDE_CONFIG_DIR` (reuse the `_skill_dir` root logic).
- `status` reports its `installed` / `not_installed` state; `remove` uses
  `npx skills remove aai-cli --global`.

**Open implementation detail (must verify, not guess):** the existing skill is
its own dedicated repo (`AssemblyAI/assemblyai-skill`), so `npx skills add
AssemblyAI/assemblyai-skill` maps repo → one skill. This repo hosts the skill
under `skills/aai-cli/`, a *subdirectory*. The implementer MUST confirm the
exact `skills` CLI invocation that resolves the `aai-cli` skill out of this repo
(candidates: `npx skills add AssemblyAI/cli`, `npx skills add AssemblyAI/cli
aai-cli`, or a path-qualified form) and confirm the resulting on-disk path,
adjusting the detection logic to match. If the `skills` tool cannot target a
subdirectory skill, fall back to bundling + copy (design Approach B) and note
the deviation.

## Verification

- `./scripts/check.sh` is green, including:
  - **markdownlint** over the new `skills/` markdown (the root `skills/` tree is
    not part of the generated-`docs/` lint exclusion, so it is linted — the new
    files must pass).
  - **pytest with the 90% branch-coverage gate** — extend the `claude.py` tests
    so `install` / `status` / `remove` now assert **three** steps (MCP +
    `assemblyai` skill + `aai-cli` skill), covering the `npx`-missing skip path
    and the `--force` / already-installed branches for the new step.
- Every command/flag example in the skill matches real `aai <cmd> --help`.
- Security guarantees untouched: no API key on disk or argv; env↔credential
  binding unchanged (this work only adds docs + an install step).

## Distribution-readiness risk (flag, do not solve here)

`npx skills add AssemblyAI/cli` requires `github.com/AssemblyAI/cli` to be
publicly reachable. If the repo is private, external users' installs fail. Flag
this to the maintainer as a release-readiness item; it is out of scope for the
skill implementation itself.
