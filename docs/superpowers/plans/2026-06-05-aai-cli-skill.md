# aai-cli skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a model-invocable `aai-cli` skill (mirroring `vercel/skills/vercel-cli`) that teaches an agent how to drive the installed `aai` CLI safely, and wire it into `aai claude install`/`status`/`remove`.

**Architecture:** A new repo-root `skills/aai-cli/` directory with a lean `SKILL.md` (decision tree + the auth/env/output gotchas + anti-patterns) routing to four `references/*.md` files grouped to match the CLI's own `aai --help` sections. Distribution mirrors the existing `assemblyai` skill step in `aai_cli/commands/claude.py`: a third `npx skills add` step, detected on disk under `~/.claude/skills/aai-cli/`.

**Tech Stack:** Markdown (skill content), Python/Typer (`aai_cli/commands/claude.py`), pytest (coverage gate 90%), ruff/mypy/markdownlint via `./scripts/check.sh`.

**Spec:** `docs/superpowers/specs/2026-06-05-aai-cli-skill-design.md`

---

## File Structure

- **Create** `skills/aai-cli/SKILL.md` — model-invocable keystone: intro, auth & env rules, output contract, quick start, decision tree, anti-patterns.
- **Create** `skills/aai-cli/references/transcription.md` — `transcribe`, `stream`, `agent`, `llm`.
- **Create** `skills/aai-cli/references/history.md` — `transcripts`, `sessions`.
- **Create** `skills/aai-cli/references/account.md` — `login`, `logout`, `whoami`, `balance`, `usage`, `limits`, `keys`, `audit`.
- **Create** `skills/aai-cli/references/setup.md` — `init`, `samples`, `doctor`, `claude`, `version`.
- **Modify** `aai_cli/commands/claude.py` — add the `aai-cli` skill install/status/remove step.
- **Modify** `tests/test_claude.py` (or the existing claude-command test file) — assert three steps and cover the new step's branches.

### Important: capturing `--help` cleanly

In this dev environment, `uv run aai <cmd> --help` prints build/dependency noise that is **not** part of the help. Ignore these lines when transcribing content into the skill:

```
   Building aai-cli @ file:///...
      Built aai-cli @ file:///...
Uninstalled 1 package in 1ms
Installed 1 package in 1ms
Fetching malware lists from https://malware-list.aikido.dev (default)
```

Only the `Usage: ...` block, `Options`, and `Examples` are real help content.

---

## Task 1: Create `skills/aai-cli/SKILL.md`

**Files:**
- Create: `skills/aai-cli/SKILL.md`

- [ ] **Step 1: Write the file with exactly this content**

````markdown
---
name: aai-cli
description: Use the AssemblyAI CLI (`aai`) from the command line — transcribe audio/video files, URLs, and YouTube links; stream live real-time transcription from a mic/file/system audio; run full-duplex voice agents; query the LLM Gateway over transcripts; browse transcript and streaming-session history; sign in and manage account balance, usage, rate limits, API keys, and audit logs; scaffold starter apps and SDK samples (init/samples); diagnose setup (doctor); and wire up Claude Code (claude). Use whenever an agent is invoking the `aai` command.
---

# AssemblyAI CLI (`aai`)

`aai` runs AssemblyAI from the terminal: transcription, real-time streaming,
voice agents, the LLM Gateway, history, and account management.

**`aai <command> --help` is the source of truth for flags.** This skill covers
the command map and the non-obvious operational rules; check `--help` before
guessing a flag.

## Critical: auth & environment

**Authentication.** A command needs a key resolved in this order:

1. `ASSEMBLYAI_API_KEY` environment variable
2. The OS keyring (populated by `aai login`)

Get authenticated with either `aai login` (browser sign-in; stores a key in the
keyring) or by exporting `ASSEMBLYAI_API_KEY`. **Run commands deliberately have
no `--api-key` flag** — that is on purpose, so keys never land in `ps` output or
shell history. Do not look for one.

**Environment binding.** The backend environment is selected by `--env`
(or `AAI_ENV`, or the profile's stored env). `--sandbox` is shorthand for
`--env sandbox000`. The default environment is currently `sandbox000`.
**A credential is only valid against the environment that minted it** — a
sandbox key fails against production and vice-versa. If a freshly-working key
suddenly returns auth errors, check you are on the same `--env` you logged in
under.

**Profiles.** `--profile <name>` selects a named credential set. Global flags
(`--profile`, `--env`, `--sandbox`) go *before* the subcommand:
`aai --sandbox transcribe call.mp3`.

## Output contract (read this before parsing output)

- **Data goes to stdout; errors and progress go to stderr.** Piping stdout is
  always safe.
- **`--json` is auto-enabled when output is piped or the CLI detects an agent
  run**, so you usually get machine-readable JSON on stdout for free. Pass
  `--json` explicitly to force it. Many commands also accept `-o/--output` to
  print a single field (e.g. `-o text`).
- Expected failures print a clean message to stderr and exit non-zero — never a
  traceback. Exit code reflects the error type.

## Quick start

```bash
aai login                      # browser sign-in (or: export ASSEMBLYAI_API_KEY=...)
aai doctor                     # verify the environment is ready
aai transcribe call.mp3        # transcribe a file
aai transcribe call.mp3 -o text   # just the text, pipeline-friendly
aai stream                     # live transcription from the mic
aai init                       # scaffold a starter app
```

## Decision tree

- **Transcribe a file/URL/YouTube, stream live audio, run a voice agent, or
  query the LLM Gateway** → `references/transcription.md`
- **Browse past transcripts or streaming sessions** → `references/history.md`
- **Sign in/out, identity, balance, usage, rate limits, API keys, audit log** →
  `references/account.md`
- **Scaffold apps/samples (`init`, `samples`), diagnose setup (`doctor`), wire
  up Claude Code (`claude`), version** → `references/setup.md`

## Anti-patterns

- **Passing `--api-key` to a run command.** It does not exist. Use `aai login`
  or `ASSEMBLYAI_API_KEY`.
- **Mixing a credential with the wrong `--env`.** A `sandbox000` key won't work
  against production. Log in and run under the same environment.
- **Running before authenticating.** No key → auth failure. Run `aai doctor` to
  see exactly what's missing.
- **Assuming `pip install assemblyai-cli` works.** That PyPI name is squatted by
  an unrelated third party. Use the project's official install path, not that
  name.
- **Parsing human output.** Pipe stdout (auto-JSON) or pass `--json` / `-o text`
  rather than scraping the pretty-printed tables.
- **Forgetting `--show-code`.** `transcribe`, `stream`, and `agent` accept
  `--show-code` to print a ready-to-run Python SDK script for exactly the flags
  you passed — no API call made. Great for "how would I do this in code?".
````

- [ ] **Step 2: Lint the file**

Run: `uv run --with-requirements /dev/null markdownlint skills/aai-cli/SKILL.md` — or simply rely on the project's markdownlint via `./scripts/check.sh` in Task 6. If markdownlint is installed locally:

Run: `markdownlint skills/aai-cli/SKILL.md`
Expected: no output (clean), or fix any reported line-length/heading issues.

- [ ] **Step 3: Commit**

```bash
git add skills/aai-cli/SKILL.md
git commit -m "Add aai-cli skill SKILL.md"
```

---

## Task 2: Create `references/transcription.md` (the exemplar)

This file is written **in full** below — it is the worked example whose
structure Tasks 3 covers for the other groups. Content is verified against
`aai transcribe|stream|agent|llm --help`.

**Files:**
- Create: `skills/aai-cli/references/transcription.md`

- [ ] **Step 1: Write the file with exactly this content**

````markdown
# Transcription & AI

Four commands. All accept `--json` (auto-enabled when piped) and `-o/--output`
to print a single field. `transcribe`, `stream`, and `agent` accept
`--show-code` to print equivalent Python SDK code without calling the API.

## `aai transcribe [SOURCE]` — file / URL / YouTube

`SOURCE` is a local file path, public URL, or YouTube URL (downloaded first).
Use `--sample` for the hosted `wildfires.mp3`. Analysis results (summary,
chapters, sentiment, …) render automatically in human mode.

High-value flags (run `aai transcribe --help` for the full set):

- Model/language: `--speech-model` (best, nano, slam-1, universal),
  `--language-code en_us`, `--language-detection`.
- Diarization: `--speaker-labels`, `--speakers-expected N`, `--multichannel`.
- PII: `--redact-pii`, `--redact-pii-policy person_name,...`,
  `--redact-pii-sub hash|entity_name`, `--redact-pii-audio`.
- Audio intelligence: `--summarization`, `--auto-chapters`,
  `--sentiment-analysis`, `--entity-detection`, `--auto-highlights`,
  `--topic-detection`, `--content-safety`.
- Escape hatch to any SDK field: `--config KEY=VALUE` (repeatable) and
  `--config-file config.json`.
- Post-process: `--llm "PROMPT"` (repeatable; chains over the transcript via LLM
  Gateway), `--translate-to es` (repeatable).
- Output: `-o text|id|status|utterances|srt|json`, `--json`, `--show-code`.

Examples:

```bash
aai transcribe call.mp3
aai transcribe --sample
aai transcribe call.mp3 --speaker-labels --speakers-expected 2 --redact-pii
aai transcribe call.mp3 -o text
aai transcribe call.mp3 --show-code
```

## `aai stream [SOURCE]` — live real-time transcription

Omit `SOURCE` to use the microphone; pass a file/URL/YouTube to stream that, or
`--sample`. macOS can capture system audio with `--system-audio` (mic + system)
or `--system-audio-only`.

High-value flags (run `aai stream --help` for the full set):

- Capture: `--device N`, `--sample-rate HZ`, `--encoding pcm_s16le|pcm_mulaw`.
- Model/turns: `--speech-model` (default `u3-rt-pro`), `--format-turns`,
  `--include-partial-turns`, `--end-of-turn-confidence`, `--min-turn-silence`,
  `--max-turn-silence`, `--vad-threshold`.
- Features: `--speaker-labels`, `--max-speakers`, `--keyterms-prompt`,
  `--redact-pii`, `--voice-focus near_field|far_field`, `--domain medical`.
- Live LLM: `--llm "PROMPT"` (refreshes the answer on every finalized turn).
- Output: `-o text|json`, `--json` (newline-delimited JSON events),
  `--show-code`.

Examples:

```bash
aai stream
aai stream --system-audio
aai stream --sample
aai stream --llm "summarize action items"
aai stream -o text                 # finalized turns as plain lines, pipe-friendly
```

## `aai agent [SOURCE]` — full-duplex voice agent

Two-way voice conversation (mic in, TTS out). Pass a file/URL or `--sample` to
speak a recorded clip instead of the mic; the session then ends after the reply.

High-value flags:

- `--voice ivy` (see `--list-voices`), `--system-prompt "..."` or
  `--system-prompt-file path`, `--greeting "..."`, `--device N`.
- Output: `-o text|json`, `--json`, `--show-code`.

Examples:

```bash
aai agent
aai agent --voice james --greeting "Hi there"
aai agent --list-voices
aai agent --show-code
```

## `aai llm [PROMPT]` — LLM Gateway

Send a prompt to the LLM Gateway. With `--transcript-id ID` the transcript's
text is injected server-side so you can ask questions about a past
transcription. Reads stdin when piped.

High-value flags:

- `--model` (default `claude-haiku-4-5-20251001`, see `--list-models`),
  `--transcript-id ID`, `--system "..."`, `--max-tokens N`.
- `-f/--follow`: re-run the prompt over a transcript growing on stdin,
  refreshing the answer in place on every finalized turn.
- Output: `-o text|json`, `--json`.

Examples:

```bash
aai llm "summarize" --transcript-id 5551234-abcd
echo "meeting notes" | aai llm "turn into action items"
aai stream -o text | aai llm -f "summarize action items as I talk"
aai llm --list-models
```
````

- [ ] **Step 2: Commit**

```bash
git add skills/aai-cli/references/transcription.md
git commit -m "Add aai-cli transcription reference"
```

---

## Task 3: Create `history.md`, `account.md`, `setup.md`

These three reference files follow the **same structure as `transcription.md`**
(Task 2): for each command — a `##` heading with its usage line, one sentence of
purpose, a short bullet list of the highest-value flags, and a fenced
`Examples` block copied from the command's own `--help` Examples section.

**Procedure for each command below:** run `aai <command> --help`, ignore the
build/aikido noise (see "Important" at the top of this plan), and transcribe its
purpose + top flags + the `Examples` block.

**Files:**
- Create: `skills/aai-cli/references/history.md`
- Create: `skills/aai-cli/references/account.md`
- Create: `skills/aai-cli/references/setup.md`

- [ ] **Step 1: Gather help for the History group**

Run: `uv run aai transcripts --help && uv run aai sessions --help`
(Note: `transcripts` and `sessions` are sub-apps — also check their subcommands,
e.g. `aai transcripts --help` lists `list`/`get`/etc.; run `--help` on each.)

- [ ] **Step 2: Write `references/history.md`**

Header `# History`, then a `##` section per command (`transcripts`, `sessions`
and their subcommands). Each section: usage line, one-sentence purpose, top
flags as bullets, and an `Examples` fenced block from `--help`. Emphasize that
output is JSON-friendly (pipe stdout / `--json`) so an agent can parse listings
and fetch a transcript by id, and cross-link: "fetch a transcript's text for a
prompt with `aai llm --transcript-id` (see `transcription.md`)."

- [ ] **Step 3: Gather help for the Account group**

Run each: `uv run aai login --help`, `... logout --help`, `... whoami --help`,
`... balance --help`, `... usage --help`, `... limits --help`,
`... keys --help` (a sub-app — also its subcommands), `... audit --help`.

- [ ] **Step 4: Write `references/account.md`**

Header `# Account`, then a `##` section per command. Cross-reference the
SKILL.md auth/env rules rather than repeating them: note that `login` stores the
key in the keyring bound to the active `--env`, `whoami` reports whether the
active profile's key is usable, and `keys` manages AssemblyAI API keys (list/
create/rename). Include each command's `Examples` block.

- [ ] **Step 5: Gather help for the Setup group**

Run each: `uv run aai init --help`, `... samples --help`, `... doctor --help`,
`... claude --help` (and `aai claude install|status|remove --help`),
`... version --help`.

- [ ] **Step 6: Write `references/setup.md`**

Header `# Setup & Tools`, then a `##` section per command. For `init` note the
templates (`audio-transcription` / `live-captions` / `voice-agent`) and that it
writes the key to a git-ignored `.env`; for `claude` note it wires the
`assemblyai-docs` MCP + the `assemblyai` skill **and this `aai-cli` skill** (see
Task 4) into Claude Code. Include each command's `Examples` block.

- [ ] **Step 7: Commit**

```bash
git add skills/aai-cli/references/history.md skills/aai-cli/references/account.md skills/aai-cli/references/setup.md
git commit -m "Add aai-cli history, account, and setup references"
```

---

## Task 4: Wire the skill into `aai claude install`/`status`/`remove`

Mirror the existing `_install_skill` / `_skill_status` / `_remove_skill`
functions in `aai_cli/commands/claude.py` for a second skill named `aai-cli`
sourced from **this** repo.

**Files:**
- Modify: `aai_cli/commands/claude.py`
- Test: `tests/` (the file that tests `claude.py` — find it in Task 5)

- [ ] **Step 1: VERIFY the exact `skills` invocation (do not guess)**

The existing skill is a dedicated repo (`AssemblyAI/assemblyai-skill` → skill
`assemblyai`). This skill lives in a **subdirectory** (`skills/aai-cli/`) of
`AssemblyAI/cli`. Confirm how the `skills` CLI targets it:

Run: `npx -y skills add --help` and `npx -y skills --help`

Determine which form installs **only** the `aai-cli` skill into
`~/.claude/skills/aai-cli/` (candidates, in order of likelihood):
`npx skills add AssemblyAI/cli aai-cli`, `npx skills add AssemblyAI/cli`,
or a path-qualified form. Verify by running it and checking that
`~/.claude/skills/aai-cli/SKILL.md` appears (and that the repo's `.claude/skills`
dev skills are NOT pulled in). Use the confirmed command in Step 2.

**Fallback:** if no `skills` form can target the subdirectory skill, stop and
switch to spec Approach B (bundle the skill in the wheel via
`[tool.hatch.build.targets.wheel] artifacts` and copy it into
`~/.claude/skills/aai-cli` on install). Note the deviation in the commit.

- [ ] **Step 2: Add the constants and helpers**

Near the existing `_SKILL_ADD` block in `aai_cli/commands/claude.py`, add (using
the command form confirmed in Step 1 — shown here with the most likely form):

```python
CLI_SKILL_REPO = "AssemblyAI/cli"
_CLI_SKILL_NAME = "aai-cli"
_CLI_SKILL_ADD = ["npx", "-y", "skills", "add", CLI_SKILL_REPO, _CLI_SKILL_NAME, "--global", "--yes"]
_CLI_SKILL_REMOVE = ["npx", "-y", "skills", "remove", _CLI_SKILL_NAME, "--global"]
_CLI_SKILL_ADD_HINT = f"npx skills add {CLI_SKILL_REPO} {_CLI_SKILL_NAME} --global"


def _cli_skill_dir() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    root = Path(config_dir) if config_dir else Path.home() / ".claude"
    return root / "skills" / _CLI_SKILL_NAME


def _cli_skill_installed() -> bool:
    return (_cli_skill_dir() / "SKILL.md").exists()


def _install_cli_skill(force: bool) -> Step:
    if shutil.which("npx") is None:
        return {
            "name": "aai-cli skill",
            "status": "skipped",
            "detail": f"Node.js/npx not found. Install Node.js, then run: {_CLI_SKILL_ADD_HINT}",
        }
    if _cli_skill_installed() and not force:
        return {
            "name": "aai-cli skill",
            "status": "already",
            "detail": f"aai-cli skill at {_cli_skill_dir()}",
        }
    proc = _run(_CLI_SKILL_ADD, timeout=300)
    if proc.returncode != 0:
        return {"name": "aai-cli skill", "status": "failed", "detail": _proc_detail(proc)}
    if not _cli_skill_installed():
        return {
            "name": "aai-cli skill",
            "status": "failed",
            "detail": (
                f"'{' '.join(_CLI_SKILL_ADD[3:])}' reported success but no skill was found at "
                f"{_cli_skill_dir()}. Install it manually: {_CLI_SKILL_ADD_HINT}"
            ),
        }
    return {"name": "aai-cli skill", "status": "installed", "detail": str(_cli_skill_dir())}


def _cli_skill_status() -> Step:
    return {
        "name": "aai-cli skill",
        "status": "installed" if _cli_skill_installed() else "not_installed",
        "detail": str(_cli_skill_dir()),
    }


def _remove_cli_skill() -> Step:
    if not _cli_skill_installed():
        return {"name": "aai-cli skill", "status": "not_installed", "detail": str(_cli_skill_dir())}
    if shutil.which("npx") is None:
        return {
            "name": "aai-cli skill",
            "status": "skipped",
            "detail": f"Node.js/npx not found. Remove manually: {' '.join(_CLI_SKILL_REMOVE)}",
        }
    proc = _run(_CLI_SKILL_REMOVE, timeout=120)
    if proc.returncode != 0 or _cli_skill_installed():
        detail = _proc_detail(proc) or "skill still present after removal"
        return {"name": "aai-cli skill", "status": "failed", "detail": detail}
    return {"name": "aai-cli skill", "status": "removed", "detail": str(_cli_skill_dir())}
```

Also rename the existing `assemblyai` skill step's `"name"` from `"skill"` to a
clearer label if desired, but **keep it `"skill"`** to avoid churning existing
tests — only the new step uses `"aai-cli skill"`.

- [ ] **Step 3: Add the new step to the three command bodies**

In `install`'s `body`, change the steps list to include the new step:

```python
steps = [_install_mcp(scope, force), _install_skill(force), _install_cli_skill(force)]
```

In `status`'s `body`:

```python
steps = [_mcp_status(), _skill_status(), _cli_skill_status()]
```

In `remove`'s `body`:

```python
steps = [_remove_mcp(scope), _remove_skill(), _remove_cli_skill()]
```

- [ ] **Step 4: Type-check and lint**

Run: `uv run mypy && uv run ruff check aai_cli/commands/claude.py`
Expected: clean. (Watch for the ruff PostToolUse autofix stripping a
now-unused import; re-add if needed.)

- [ ] **Step 5: Commit**

```bash
git add aai_cli/commands/claude.py
git commit -m "Install the aai-cli skill alongside the docs MCP and assemblyai skill"
```

---

## Task 5: Tests for the new install step

**Files:**
- Test: locate with `grep -rl "_install_skill\|claude install\|MCP_NAME\|assemblyai-docs" tests/`

- [ ] **Step 1: Find and read the existing claude tests**

Run: `grep -rl "assemblyai-docs\|_install_skill\|claude" tests/`
Read the file. Note how it stubs `shutil.which`, `_run`, and `_skill_installed`
(likely via monkeypatch) and how it asserts on the steps list / rendered output.

- [ ] **Step 2: Write failing tests for the three branches of the new step**

Mirror the existing `assemblyai` skill tests. Add tests asserting:

```python
def test_install_includes_aai_cli_skill(monkeypatch, ...):
    # npx present, skill not yet installed, _run returns success, dir then exists
    # -> steps include {"name": "aai-cli skill", "status": "installed", ...}
    ...

def test_install_aai_cli_skill_skipped_without_npx(monkeypatch, ...):
    # shutil.which("npx") -> None
    # -> {"name": "aai-cli skill", "status": "skipped", ...}
    ...

def test_status_reports_aai_cli_skill(monkeypatch, ...):
    # _cli_skill_installed() patched True/False
    # -> status step present with installed / not_installed
    ...

def test_remove_aai_cli_skill(monkeypatch, ...):
    # installed then removed; assert {"name": "aai-cli skill", "status": "removed"}
    ...
```

Match the existing tests' exact monkeypatch targets (e.g.
`aai_cli.commands.claude._run`, `...shutil.which`, `..._cli_skill_installed`).
Update any existing test that asserts the **number** of steps (was 2, now 3) or
asserts the full steps list.

- [ ] **Step 3: Run the new tests to verify they fail (before Step 2 of Task 4 if done out of order) / pass now**

Run: `uv run pytest tests/<claude_test_file>.py -q`
Expected: PASS (Task 4 already added the implementation). If you wrote tests
first, they fail with the expected missing-step assertion, then pass after Task 4.

- [ ] **Step 4: Commit**

```bash
git add tests/<claude_test_file>.py
git commit -m "Test the aai-cli skill install/status/remove step"
```

---

## Task 6: Full gate + content verification

**Files:** none (verification only)

- [ ] **Step 1: Verify every skill example against real `--help`**

For each command referenced in the skill, confirm the flags exist:

Run: `uv run aai transcribe --help`, `... stream --help`, `... agent --help`,
`... llm --help`, and the History/Account/Setup commands. Cross-check each flag
named in the reference files against its `--help`. Fix any drift.

- [ ] **Step 2: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` This runs ruff, ruff format --check,
mypy, **markdownlint** (which now lints `skills/aai-cli/**.md` — fix any
line-length/heading/list issues), shellcheck, pytest with the **90%
branch-coverage gate**, and build + `twine check --strict`.

If markdownlint flags the new markdown, fix the files and re-run until green.

- [ ] **Step 3: Manual smoke test of the install wiring**

Run: `uv run aai claude status --json`
Expected: JSON listing three steps including `"aai-cli skill"`. (Install/remove
need network + npx; run `aai claude install` manually if you want a live check,
then `aai claude remove`.)

- [ ] **Step 4: Run the project review skill**

Run `/review-changes` on the diff (it runs the `code-review` skill; this diff
touches `aai_cli/commands/claude.py` which shells out via `subprocess`, so the
`security-review` + `security-reviewer` agent paths apply). Address findings.

- [ ] **Step 5: Final commit if review produced changes**

```bash
git add -A && git commit -m "Address review feedback for aai-cli skill"
```

---

## Self-review notes

- **Spec coverage:** SKILL.md (Task 1) ✓; references mirroring CLI `--help`
  groups (Tasks 2–3) ✓; auth/env/output/anti-patterns content (Task 1) ✓;
  `claude install`/`status`/`remove` wiring (Task 4) ✓; tests for 90% gate
  (Task 5) ✓; markdownlint over `skills/` + example verification + check.sh
  (Task 6) ✓; distribution-readiness risk + `npx` subdir verification
  (Task 4 Step 1) ✓.
- **`disable-model-invocation` intentionally omitted** from SKILL.md frontmatter
  so the skill is model-invocable (unlike the repo's dev skills) — this is the
  whole point and matches the spec.
- **Distribution-readiness risk** (repo must be public for external `npx skills
  add`) is a maintainer flag from the spec, surfaced in Task 4 Step 1's fallback
  rather than solved here.
