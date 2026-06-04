# `aai init` — interactive project scaffolder

**Status:** Approved (design)
**Date:** 2026-06-04

## Summary

Add an `aai init` command: a Vercel-style scaffolder that lets a user pick from
four templates, then **copies a real, runnable Python web project**, installs its
dependencies, starts a local server, and opens the browser — so the user can
interact with an AssemblyAI capability immediately and then keep building.

The generated project's prime directive: **the files must be as simple as
possible** so a human (or Claude Code) can read and extend them right after
creation. No build step, no framework magic — two files carry the real logic.

## Goals

- One command (`aai init`) takes a user from nothing to a running, browser-based
  AssemblyAI demo.
- Four templates covering the CLI's core capabilities: `transcribe`, `stream`,
  `agent`, `llm`.
- Generated code is flat, plain, and immediately editable.
- The browser **never** receives the real API key.
- Each template is **structured so Vercel can deploy it as-is** (static frontend +
  a thin stateless backend). We add the conventional layout + a `vercel.json`, but
  build **no** deploy tooling of our own (no Dockerfiles, deploy buttons, or
  Cloudflare functions) — that would only overlap with what Vercel already does.

## Non-goals

- Replacing `aai samples create` (it coexists — see below).
- A Node/JavaScript toolchain. Templates are Python backend + a single vanilla
  HTML/JS page.
- A reusable framework or component library. Each template is self-contained.

## Relationship to existing commands

- **`aai samples create`** stays as-is: the quick **single-script** scaffolder
  (one `.py` file, terminal output, no browser). `aai init` is the **full
  interactive project + browser** experience. Clear split: *samples = snippet,
  init = app.*
- Template `api/index.py` logic is kept consistent with `aai_cli/code_gen`'s
  generators (same AssemblyAI call patterns), verified by a test. `code_gen`
  remains the reference; templates are committed real files (Approach A below).

## Architecture decision: ship real template projects (Approach A)

The four templates live as **actual, committed, runnable project directories**
inside the package (`aai_cli/init/templates/<id>/`). `aai init` *copies* the
chosen one rather than rendering it from string templates.

- The generated output **is** the files we hand-wrote and tested — no
  template-rendering indirection between what we verify and what the user gets.
  This is the simplest possible output, which is the core constraint.
- Templates are runnable in-repo, so they are easy to test and dogfood.
- Trade-off: some AssemblyAI call logic overlaps with `code_gen`. We keep them
  honest with a smoke test that loads each template and an optional
  reference-consistency check.

(Rejected: generating everything from `code_gen` + a web shell. String
templating produces more abstract/indirected code — the opposite of the
"simple files Claude can improve" goal.)

## Command surface

```
aai init [TEMPLATE] [DIRECTORY] [options]
```

- **No args** → arrow-key picker (`questionary`) for the template, then a prompt
  for the directory (default `./<template>-app`).
- **TEMPLATE** ids mirror the CLI commands for memorability: `transcribe`,
  `stream`, `agent`, `llm`. Descriptive titles shown in the picker.
- **Options:**
  - `--no-install` — scaffold only; skip env creation, install, launch, open.
  - `--no-open` — install + launch, but don't open the browser.
  - `--force` — overwrite a non-empty target directory.
  - `--here` — scaffold into the current directory.
  - `--port` — server port (default `3000`; auto-increments if taken).
  - `--json` — machine-readable step output.

**Default behavior (no skip flags):** scaffold → install deps → launch server →
open browser. Vercel-style "it just opens."

## Scaffold layout (per template)

Templates adopt **Vercel's conventional layout** so the exact same files run
locally (via `uvicorn`) and deploy to Vercel untouched — see Deployment below.

```
<dir>/
  api/index.py      # FastAPI ASGI app: loads key, mints temp tokens (+ upload/poll for transcribe & llm)
  index.html        # one vanilla-JS page, no build step
  vercel.json       # routes "/" to index.html and "/api/*" to the FastAPI function
  requirements.txt  # fastapi, assemblyai, python-dotenv (uvicorn for local run)
  .env              # ASSEMBLYAI_API_KEY=…   (written by init, gitignored)
  .env.example      # ASSEMBLYAI_API_KEY=
  .gitignore        # .env, .venv, __pycache__
  README.md         # what it is, how to run, deploy-to-Vercel note, "ideas to extend" hints
```

Only `api/index.py` and `index.html` carry real logic; the rest is boilerplate.

## Template behavior

Two backend shapes. **`stream` and `agent` are architecturally identical**: the
backend only mints a short-lived temp token and the **browser holds the WebSocket
directly to AssemblyAI** — no audio ever passes through the backend.
**`transcribe` and `llm`** need a thin backend proxy (file upload + transcript
polling) because the REST/upload endpoints aren't temp-token authorized.

- **transcribe** — drop/pick an audio or video file → backend uploads to
  AssemblyAI, creates the transcript, and the browser polls a short `/api/status`
  call until done → page renders the JSON (speaker labels, chapters, sentiment,
  entities, highlights). The split into create + poll keeps each backend call
  short, which is also what makes it serverless-friendly.
- **stream** (live captions) — browser captures mic audio and connects to
  AssemblyAI Streaming v3 **directly using a short-lived temp token** minted by
  the backend `/api/token` route; renders partial → final turns live.
- **agent** (voice agent) — same pattern as `stream`: backend `/api/token` mints
  a token (`GET /v1/token`, with `expires_in_seconds` / `max_session_duration_seconds`),
  and the browser opens its own WebSocket directly to
  `wss://agents.assemblyai.com/v1/ws?token=…`, streaming mic audio up and playing
  agent audio back. No backend audio proxy, no persistent server WebSocket.
- **llm** (chat with audio) — transcribe a file (same create+poll path as the
  transcribe template), then a chat box posts questions to `/api/chat`, which
  calls the LLM Gateway over the transcript and returns answers.

## API key handling & browser safety (invariant)

- `aai init` resolves the key via the **CLI's existing resolution chain**
  (`ASSEMBLYAI_API_KEY` env → OS keyring) and writes it to the **gitignored**
  `.env`. The backend loads it with `python-dotenv`.
- **The browser never receives the real key.** File uploads and LLM calls go
  through the backend (`transcribe`/`llm`); `stream` and `agent` use the backend's
  `/api/token` endpoint, which mints a short-lived AssemblyAI temporary token for
  the browser. For every template the API key stays server-side.
- **No key resolvable** → scaffold anyway, write `.env` with a placeholder value,
  warn the user, and **skip auto-launch** (the server would fail without a key).

**Noted tension:** this is the one place the system diverges from the CLI's
"never write the key to a plaintext dotfile" rule. Justification: the scaffold is
a *separate generated project* (not the CLI's own credential store), `.env` is
gitignored, and `.env` is the universal web-dev convention. Accepted deliberately.

## Install / launch flow

Reuses the "steps with status" rendering pattern from `aai claude` (themed step
lines, `--json` support).

1. Resolve template + target dir. Conflict (non-empty dir) → error unless
   `--force`/`--here`.
2. Copy template files; write `.env` (from resolved key), `.env.example`,
   `.gitignore`.
3. Create the environment: `uv venv` if `uv` is on PATH, else stdlib
   `python -m venv`; install `requirements.txt` (`uv pip install` / `pip
   install`). *(skipped on `--no-install`)*
4. Launch the server as a subprocess — `uvicorn api.index:app --port <port>` (via
   `uv run` or the created venv) — wait for the port to accept connections, open
   the browser. *(skipped on `--no-install`/`--no-open`)*
5. Print the URL and next steps. Ctrl-C stops the server cleanly (exit 0).

## Deployment (Vercel-ready, no extra tooling)

Each template uses Vercel's conventional layout (`api/index.py` + `index.html` +
`vercel.json`), so the **same files run locally and deploy to Vercel untouched**:
the user pushes the generated directory to a Git repo, imports it on Vercel, and
sets `ASSEMBLYAI_API_KEY` as a Vercel environment variable (the local `.env` is
git-ignored and never deployed). The README includes a short "Deploy to Vercel"
note.

- All four templates are Vercel-deployable because every backend is **thin and
  stateless**: `stream`/`agent` only mint a temp token, and `transcribe`/`llm`
  split work into short create + poll calls — no persistent WebSocket server and
  no long-running request.
- We deliberately ship **no deploy tooling of our own** — no Dockerfiles, deploy
  buttons, CI config, or Cloudflare functions. That would only duplicate what
  Vercel already provides. Our responsibility ends at producing a layout Vercel
  can deploy.
- Cloudflare is explicitly out of scope (no production Python runtime; would
  require a JS rewrite).

## Error handling

- `uv` missing → fall back to stdlib `venv` + `pip` (shown in the step detail).
- Install failure → leave scaffolded files in place, print manual run steps, exit
  non-zero.
- Port in use → increment to the next free port (or honor `--port`).
- Target dir exists & non-empty → error unless `--force`.
- No key → warn, scaffold anyway, do not auto-launch.
- Ctrl-C during the running server → clean shutdown, exit 0 (mirrors the CLI's
  existing closed-pipe / signal handling posture).

## Code organization

- `aai_cli/commands/init.py` — the Typer command; registered in `_COMMAND_ORDER`
  in `main.py` (placed near `samples`).
- `aai_cli/init/` package:
  - `scaffold.py` — copy template files, write `.env`/`.env.example`/`.gitignore`.
  - `runner.py` — environment setup, dependency install, server launch, browser
    open, port selection.
  - `templates/<id>/…` — the four real template projects, shipped as package data.
- New CLI dependency: `questionary` (arrow-key picker).
- `fastapi` / `uvicorn` / `python-dotenv` are dependencies of the **generated**
  `requirements.txt`, **not** of the CLI itself.
- `pyproject.toml` — include `aai_cli/init/templates/**` (all template files,
  incl. `vercel.json`, `.env.example`, dotfiles) as package data.

## Testing (TDD)

- **Unit:** template/dir argument resolution, picker default-dir logic,
  target-dir conflict handling, key → `.env` writing (assert the committed
  template fixtures contain **no** real key), step rendering, `--json` output.
- **Template integrity:** each template loads via FastAPI `TestClient`; `GET /`
  returns 200 and serves the page; `/api/token` mints a token (AssemblyAI mocked)
  for `stream`/`agent`. Each template directory contains the required files
  (incl. `vercel.json`).
- **Install flow:** with **mocked** subprocess — assert correct `uv`/`venv`/`pip`
  commands, the uv→venv fallback path, and that `--no-install`/`--no-open` skip
  their steps. No real network or pip in tests.
- **Reference consistency (light):** each template's core AssemblyAI call matches
  the corresponding `code_gen` reference pattern.

## Open questions

None blocking. Future follow-ups (out of scope for v1): additional templates,
a `--template-url` for community templates, and first-class Cloudflare support
(would require JS-based edge functions).
