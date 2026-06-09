# Design: `aai onboard` — Guided Onboarding Flow

**Date:** 2026-06-08
**Status:** Approved design, pending implementation plan
**Goal:** Make the `aai` CLI the easiest way to onboard to AssemblyAI and reach
100 API requests. Attack all four funnel stages: install→first request,
first request→habit, account/billing friction, and discovery of value.

## Background

Today the CLI's onboarding is a set of discrete, well-built commands (`login`,
`init`, `doctor`, `samples`, `setup`) with **no unifying first-run flow**. A
newcomer must already know the sequence. There is also **no visibility into
progress toward a usage milestone**, so a user who stops at one request gets no
nudge to return.

The design borrows OpenClaw's onboarding blueprint
(`github.com/openclaw/openclaw`), specifically:

- A single `onboard` wizard distinct from a minimal `setup`
  (`src/commands/onboard-interactive.ts` → `runSetupWizard`).
- A **prompter abstraction** so one flow runs both interactively and
  non-interactively (`createClackPrompter` vs
  `createNonInteractiveLoggingPrompter`).
- **Ordered, resumable sections** that self-skip when already satisfied
  (`CONFIGURE_WIZARD_SECTIONS`).
- **Grouped auth choice** shared by onboarding and later setup
  (`promptAuthChoiceGrouped`).
- **Terminal restore on every exit path** (`finally` block; clean cancel).
- **Ending on a concrete next action** (OpenClaw's onboarding chat).

The one thing OpenClaw lacks that our goal demands — **progress toward a usage
milestone (100 requests)** — is our addition.

## Scope

All seven sections ship in this spec (chosen over a smaller MVP). Progress is
measured with a **local CLI request counter** (chosen over querying account
usage), because it is always available, adds zero API calls, and ships fast. A
one-line pointer to `aai usage` gives the authoritative account picture without
us owning that mapping.

## Architecture

New package `aai_cli/onboard/` plus one command module:

```
aai_cli/onboard/
  __init__.py
  wizard.py        # run_onboarding(prompter, state, ctx) — orchestrates sections in order
  sections.py      # each step: (prompter, ctx) -> SectionResult (DONE | SKIPPED | FAILED)
  prompter.py      # Prompter protocol + InteractivePrompter / NonInteractivePrompter
  progress.py      # local request counter + "N of 100" rendering + milestone copy
aai_cli/commands/onboard.py   # Typer sub-app: builds prompter, runs wizard, restores terminal in finally
```

Conventions follow the repo: `from __future__ import annotations`, modern
typing, errors→stderr / data→stdout, strict mypy on `aai_cli`, command bodies
wrapped via `context.run_command`.

### Prompter abstraction

`Prompter` is a `typing.Protocol` with the minimal surface the sections need:

- `select(title, options) -> str`
- `confirm(title, *, default) -> bool`
- `text(title, *, default) -> str`
- `note(message)` — informational, no input
- `section(title)` — visual step header

`InteractivePrompter` wraps the existing Rich/Typer prompt helpers in
`output.py`. `NonInteractivePrompter` **never blocks for input**: `confirm`
returns its default, `select`/`text` return the provided default or raise a
clean `UsageError` when no default exists, and every call logs what it chose to
stderr. This mirrors `createNonInteractiveLoggingPrompter` and preserves the
CLI's pipeline-safety: `--json`, piped stdin, or agent-run sessions never hang.

Prompter selection reuses the existing "is this interactive?" signal already
used to auto-enable `--json` (piped/agent detection in `context.py`/`output.py`).

### Resumable sections

Each section is a function returning a `SectionResult`. Before doing work it
checks whether the step is already satisfied and returns `SKIPPED` if so:

- Auth → key already resolves (`config.resolve_api_key` succeeds).
- Environment → checks already pass.
- Build-path scaffold → target dir already exists (offer reuse/skip).
- Claude Code → `aai setup status` reports installed.

Re-running `aai onboard` therefore resumes rather than restarts. A `FAILED`
section records the reason, prints the standard `Error:`/`Suggestion:` pair, and
the wizard continues to subsequent independent sections where sensible (auth
failure is the one hard stop, since later steps need a key).

### Terminal restore

`run_onboarding` wraps the section loop in `try/finally`. `KeyboardInterrupt`
and a `WizardCancelled` sentinel both exit cleanly (no traceback, terminal
state restored). This matches the repo's existing discipline of never dumping
tracebacks for expected control flow.

## The flow (ordered sections)

1. **Welcome** — one-line value statement and a list of what the wizard will
   do. If a local progress counter already shows prior requests, greet as a
   returning user and show "N of 100" instead of the cold intro.

2. **Auth** *(install → first request gate)* — reuse
   `auth.flow.persist_browser_login()`. Offer an API-key fallback (grouped
   choice: *Browser sign-in* / *Paste an API key*). If org discovery returns
   **no account or no project**, print the signup/dashboard URL
   (`environments.signup_url()`), wait for the user, then retry. Auth is the one
   hard-stop section.

3. **First request — the activation moment** — the wizard itself runs the
   equivalent of `transcribe --sample` (hosted `wildfires.mp3`,
   `client.SAMPLE_AUDIO_URL`), streams the transcript, and celebrates success.
   This guarantees nobody completes onboarding un-activated. Increments the
   progress counter (→ "1 of 100"). On API failure, surface the normal error
   and offer retry.

4. **Environment check (non-blocking)** — run `doctor`'s existing checks
   (python / ffmpeg / mic) and render ✓/!/✗. Never a hard stop: warnings only
   matter for `stream`/`agent`. Reuse the doctor check functions rather than
   duplicating them.

5. **"What do you want to build?"** *(discovery of value + repeat requests)* —
   `select`: *Transcribe files* → `init audio-transcription`; *Live captions* →
   `init live-captions`; *Voice agent* → `init voice-agent`; *Just the CLI* →
   skip. The chosen template is scaffolded in place via the existing `init`
   path (deps install + `.env` write + browser launch already handled there).
   Offer `samples create <kind>` as a lighter-weight alternative for users who
   want a single script.

6. **Optional: wire up Claude Code** — call `aai setup install` (docs MCP +
   skills). Skipped silently if `claude`/`npx` are absent, consistent with the
   existing `setup` behavior (missing tools are reported and skipped, not
   errors).

7. **Next steps + progress** — render "✅ N of 100 API requests" and a short
   menu of copy-pasteable commands tailored to the path chosen in step 5
   (e.g. `aai transcribe <file>`, `aai stream`, `aai llm`). End on action.

## First-run autodetect

- New command **`aai onboard`**, registered in the **Quick Start** help group
  in `main.py` `_COMMAND_ORDER`, listed above `init`.
- `aai onboard --status` prints only the progress panel (section 7's counter +
  `aai usage` pointer) and exits — a cheap "where am I?" check.
- **Bare `aai` with no credentials configured** prints a short banner and
  *offers* the wizard (`Run guided setup now? [Y/n]`). It never force-hijacks
  `--help`, and with credentials present, bare `aai` behaves exactly as today.
  Non-interactive sessions get a one-line hint, not a prompt.
- `install.sh`'s final line changes from
  `"Installed. Next: run 'aai login', then 'aai transcribe --sample'."` to
  `"Installed. Next: run 'aai onboard'."` — one command to remember.
- `aai login` success hint and the `app.callback` epilog examples are updated to
  point at `aai onboard` as the canonical starting point.

## Progress toward 100 (local counter)

- Persist a small record in `config.toml` (via the existing `config.py`
  platformdirs-backed store): `requests_made: int` and `first_request_at` /
  `last_request_at` timestamps. Per-profile, alongside existing profile state.
- Increment once per successful request from the run commands: `transcribe`,
  `stream` (per session), `agent` (per session), `llm`. A single shared helper
  `progress.record_request()` is called from each command's success path so the
  logic lives in one place.
- Render "N of 100 requests" with milestone encouragement at 1, 10, 50, 100
  (e.g. first request → "You're activated 🎉"; 100 → "You've hit 100 — you're
  off the ground"). Rendering lives in `progress.py`; the wizard's final screen
  and `aai onboard --status` both call it.
- The counter is explicitly **CLI-originated requests only**. The status panel
  carries a one-line pointer: "For your full account usage, run `aai usage`."
  We do not attempt to reconcile the local count with server-side billing.

## Error handling

- All failures use the existing `CLIError` hierarchy (`errors.py`) and
  `output.emit_error` (stderr, `Error:`/`Suggestion:`), preserving exit codes.
- Auth failure inside the wizard reuses `errors.auth_failure()` / the
  `NotAuthenticated` path and stops the wizard with a retry suggestion.
- Cancellation (Ctrl-C / `WizardCancelled`) → clean exit, terminal restored, no
  traceback.
- The non-interactive prompter converts "would need to prompt" into a clean
  `UsageError` telling the user which flag/value to supply, never a hang.

## Testing

- **Prompter**: unit tests for `NonInteractivePrompter` (returns defaults,
  raises cleanly when no default, logs choices) and `InteractivePrompter`
  (drives the existing prompt helpers via injected I/O).
- **Sections**: each section tested for the DONE / SKIPPED / FAILED branches
  with a fake prompter and mocked client/auth (no real API). Auth, first-request
  (mock `client.transcribe`), env-check, build-path (mock `init`), claude-wiring
  (mock `setup`).
- **Wizard orchestration**: ordering, resume-skips-completed, auth hard-stop,
  cancellation restores terminal.
- **Progress**: counter increments once per success, persists/round-trips
  through `config.toml`, milestone copy at 1/10/50/100, `--status` rendering.
- **First-run autodetect**: bare `aai` with/without creds; interactive vs
  non-interactive banner-vs-hint.
- **Snapshot tests** (`syrupy`, `tests/__snapshots__/*.ambr`): updated
  `aai --help` ordering, new `aai onboard --help`, the wizard's rendered panels,
  and the progress/status panel. Regenerate with `--snapshot-update`; never
  hand-edit `.ambr`.
- Must clear the existing gate: 90% branch coverage, 100% patch coverage vs
  `origin/main`, no new escape hatches, strict mypy/pyright, xenon grade.

## Out of scope

- Server-side usage reconciliation / true free-tier request accounting.
- A conversational/agent-driven onboarding mode (rejected approach C).
- Changes to the auth backend (Stytch B2B discovery flow is reused as-is).
- New `init` templates.

## Open questions

None blocking. Milestone copy wording can be refined during implementation.
