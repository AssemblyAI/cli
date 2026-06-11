---
name: aai-cli
description: Use the AssemblyAI CLI (`assembly`) from the command line — transcribe audio/video files, URLs, and YouTube/podcast links; stream live real-time transcription from a mic/file/system audio; run full-duplex voice agents; query the LLM Gateway over transcripts; browse transcript and streaming-session history; sign in and manage account balance, usage, rate limits, API keys, and audit logs; scaffold a starter app (init); diagnose setup (doctor); and set up your coding agent's AssemblyAI docs MCP + skills (setup). Use whenever an agent is invoking the `assembly` command.
---

# AssemblyAI CLI (`assembly`)

`assembly` runs AssemblyAI from the terminal: transcription, real-time streaming,
voice agents, the LLM Gateway, history, and account management.

**`assembly <command> --help` is the source of truth for flags.** This skill covers
the command map and the non-obvious operational rules; check `--help` before
guessing a flag.

## Critical: auth & environment

**Authentication.** A command needs a key resolved in this order:

1. `ASSEMBLYAI_API_KEY` environment variable
2. The OS keyring (populated by `assembly login`)

Get authenticated with either `assembly login` (browser sign-in; stores a key in the
keyring) or by exporting `ASSEMBLYAI_API_KEY`. **Run commands deliberately have
no `--api-key` flag** — that is on purpose, so keys never land in `ps` output or
shell history. Do not look for one.

**Environment binding.** The backend environment is selected by `--env`
(or `AAI_ENV`, or the profile's stored env). `--sandbox` is shorthand for
`--env sandbox000`. The default environment is `production`.
**A credential is only valid against the environment that minted it** — a
sandbox key fails against production and vice-versa. If a freshly-working key
suddenly returns auth errors, check you are on the same `--env` you logged in
under.

**Profiles.** `--profile <name>` selects a named credential set. Global flags
(`--profile`, `--env`, `--sandbox`) go *before* the subcommand:
`assembly --sandbox transcribe call.mp3`.

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
assembly login                      # browser sign-in (or: export ASSEMBLYAI_API_KEY=...)
assembly doctor                     # verify the environment is ready
assembly transcribe call.mp3        # transcribe a file
assembly transcribe call.mp3 -o text   # just the text, pipeline-friendly
assembly stream                     # live transcription from the mic
assembly init                       # scaffold a starter app
```

## Building an app vs running a command

If the task is to **build/create an app or project** (a transcription app, live
captions, or a **voice agent app**), that is `assembly init` — a scaffolder that
writes a full starter project (pick the `voice-agent` template for an agent
app). The verbs `assembly transcribe`, `assembly stream`, and **`assembly agent`** are *run*
commands: they perform a one-off action in the terminal (e.g. `assembly agent` holds
a live mic conversation) and produce **no code**. When someone says "build an
agent," reach for `assembly init voice-agent`, not `assembly agent`.

## Decision tree

- **Build/scaffold an app (transcription, live captions, or a voice agent app)**
  → `assembly init` — see `references/setup.md`
- **Transcribe a file/URL/YouTube/podcast page, stream live audio, run a live
  voice agent, or query the LLM Gateway** → `references/transcription.md`
- **Browse past transcripts or streaming sessions** → `references/history.md`
- **Sign in/out, identity, balance, usage, rate limits, API keys, audit log** →
  `references/account.md`
- **Scaffold a starter app (`init`), diagnose setup (`doctor`), set up
  your coding agent's MCP + skills (`setup`)** → `references/setup.md`

## Anti-patterns

- **Passing `--api-key` to a run command.** It does not exist. Use `assembly login`
  or `ASSEMBLYAI_API_KEY`.
- **Mixing a credential with the wrong `--env`.** A `sandbox000` key won't work
  against production. Log in and run under the same environment.
- **Running before authenticating.** No key → auth failure. Run `assembly doctor` to
  see exactly what's missing.
- **Assuming `pip install assemblyai-cli` works.** That PyPI name is squatted by
  an unrelated third party. Use the project's official install path, not that
  name.
- **Parsing human output.** Pipe stdout (auto-JSON) or pass `--json` / `-o text`
  rather than scraping the pretty-printed tables.
- **Forgetting `--show-code`.** `transcribe`, `stream`, and `agent` accept
  `--show-code` to print a ready-to-run Python SDK script for exactly the flags
  you passed — no API call made. Great for "how would I do this in code?".
