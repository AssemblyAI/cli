<p align="center">
  <a href="https://www.assemblyai.com">
    <h2 align="center">AssemblyAI CLI</h2>
  </a>
</p>

<p align="center">
  Transcribe. Stream. Converse. — speech AI from your terminal.
</p>

<p align="center">
  <a href="#quick-start"><strong>Quick start</strong></a> ·
  <a href="#commands"><strong>Commands</strong></a> ·
  <a href="#pipelines"><strong>Pipelines</strong></a> ·
  <a href="https://www.assemblyai.com/docs"><strong>Docs</strong></a>
</p>

<p align="center">
  <a href="https://github.com/AssemblyAI/cli"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-D6402E?style=flat-square"></a>
  <a href="https://github.com/AssemblyAI/cli/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-D6402E?style=flat-square"></a>
  <a href="https://www.assemblyai.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-assemblyai-D6402E?style=flat-square"></a>
</p>

---

`assembly` brings [AssemblyAI](https://www.assemblyai.com) to your terminal: transcribe files, stream live audio, run a two-way voice agent, prompt the LLM Gateway, and scaffold ready-to-deploy starter apps — all pipeline-friendly, with your key kept in the OS keyring.

## Installation

### Homebrew (recommended — macOS / Linux)

```sh
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew trust assemblyai/cli
brew install assembly
```

`brew install` pulls in `ffmpeg` and `portaudio` for you, so `transcribe`, `stream`, and `agent` work out of the box. Upgrade with `brew upgrade assembly`; remove with `brew uninstall assembly`.

### pipx / uv

```sh
# pipx
pipx install "git+https://github.com/AssemblyAI/cli.git"

# uv
uv tool install "git+https://github.com/AssemblyAI/cli.git"
```

Requires Python 3.12+. Microphone and speaker support (for `stream` and `agent`) is included by default via [`sounddevice`](https://python-sounddevice.readthedocs.io) — its macOS and Windows wheels bundle PortAudio. On Linux, install the runtime once: `sudo apt-get install libportaudio2`. You'll also want [`ffmpeg`](https://ffmpeg.org) on `PATH` to decode non-WAV/URL audio.

### One-liner

```sh
curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh
```

Prefers [`pipx`](https://pipx.pypa.io), falling back to `pip --user`.

## Quick Start

```sh
assembly onboard               # guided setup: sign in, first transcription, start building
```

Prefer to do it by hand?

```sh
assembly login                 # store your API key (browser-assisted)
assembly transcribe --sample   # transcribe the hosted wildfires.mp3 sample
```

## Build An App

`assembly init` is how you **build a new app** — it copies a small, self-contained FastAPI + HTML project you can run locally and deploy to Vercel as-is, the starting point whenever you want to *create* something, including a voice agent app:

```sh
assembly init                            # pick a template, scaffold, install deps, open the browser
assembly init audio-transcription myapp  # non-interactive: template + directory
assembly init voice-agent my-agent       # build a voice agent app (full FastAPI + browser starter)
```

Your key is written to a git-ignored `.env` (never sent to the browser). Use `--no-install` to scaffold only.

> **Building a voice agent? Use `assembly init voice-agent`, not `assembly agent`.** `assembly agent` only *runs* a live mic conversation in the terminal and writes no code; `assembly init` creates the actual app.

## Commands

| Command | What it does |
| --- | --- |
| `assembly login` / `logout` / `whoami` | Manage the stored API key. |
| `assembly doctor` | Check your environment (API key, network, ffmpeg, microphone, agent tooling). |
| `assembly transcribe <file\|url>` | Transcribe a file, URL, or YouTube/podcast page URL (`--sample`, `--llm`, `--show-code`). |
| `assembly transcripts list` / `get <id>` | Browse and fetch past transcripts. |
| `assembly stream [file]` | Real-time transcription from a file or the microphone. |
| `assembly agent` | *Run* a live two-way voice conversation (to **build** a voice agent app, use `assembly init voice-agent`). |
| `assembly llm <prompt>` | Prompt the LLM Gateway (`--transcript-id`, or `--follow` for a live stream). |
| `assembly setup install` | Set up your coding agent for AssemblyAI (docs MCP + skills). |
| `assembly keys` / `balance` / `usage` / `limits` / `sessions` / `audit` | Account self-service (browser login). |

Every command prints human-readable text by default — terminal, pipe, CI, or agent alike. Add `--json` (or `-j`) for machine-readable output; it never switches on just because stdout is piped, so `assembly transcribe call.mp3 | grep hello` still gets the transcript, not a JSON blob. Errors go to **stderr**, so stdout stays clean for pipelines.

Account data lives in **top-level** commands — `assembly balance` / `usage` / `limits` / `keys` / `audit`, and `assembly login` / `logout` / `whoami` — not under an `assembly account` group.

### JSON output

`--json` is the scripting contract. The shapes are stable:

| Command | `--json` shape |
| --- | --- |
| `transcribe` / `transcripts get` | the full transcript payload (`id`, `status`, `text`, `words`, `utterances`, …) — identical for both, so a fetched transcript round-trips |
| `transcribe --llm` | `{id, status, text, transform: {model, steps: [{prompt, output}]}}` |
| `transcripts list` / `sessions list` / `keys list` | a JSON array of row objects (`[]` when empty) |
| `balance` / `usage` / `limits` / `audit` | the raw AMS payload (e.g. `balance.balance_in_cents`; `usage.usage_items[].line_items[].price` in cents) |
| `doctor` | `{ok, profile, environment, checks: [{name, status, affects, detail, fix}]}` |
| any error | `{"error": {"type", "message", "suggestion"?, "transcript_id"?}}` on **stderr** |

`stream`/`agent` with `--json` emit newline-delimited JSON (one object per event/turn).

### Exit codes

Scripts can branch on the exit code:

| Code | Meaning |
| --- | --- |
| `0` | success |
| `1` | API/network error, missing dependency, or unexpected internal error |
| `2` | usage/validation error (bad flag, bad path, malformed id, unusable config) |
| `4` | not authenticated (no usable key, rejected key, or a self-service command needing browser login) |
| `130` | cancelled with Ctrl-C |

`assembly deploy` / `assembly dev` shell out to other tools and propagate that tool's own exit code.

> **Tip:** Quote URLs that contain `?` (most YouTube links do) — in zsh the `?` is a glob character: `assembly transcribe "https://www.youtube.com/watch?v=VIDEO_ID"`.

## Transcribe A File

`assembly transcribe` exposes the full `TranscriptionConfig` surface as curated, grouped flags — model & language, formatting, speakers & channels, PII/safety guardrails, analysis (summary, chapters, sentiment, entities, topics, highlights), customization, and webhooks:

```sh
assembly transcribe call.mp3 \
  --speaker-labels --speakers-expected 2 \
  --redact-pii --redact-pii-policy person_name,phone_number \
  --summarization --summary-type bullets \
  --sentiment-analysis --auto-chapters \
  --config speech_threshold=0.5 \
  --config-file extra.json
```

Anything without a curated flag is reachable via the escape hatch: `--config KEY=VALUE` (repeatable) and `--config-file FILE` (a JSON object) accept any SDK field by name. Precedence: config file < `--config` < explicit flags. Run `assembly transcribe --help` for the full flag list.

## Stream Live Audio

```sh
assembly stream --sample            # stream the hosted wildfires.mp3 sample
assembly stream path/to/audio.wav   # 16 kHz mono WAV streams directly (other formats need ffmpeg)
assembly stream https://…/clip.mp3  # a URL works too (decoded via ffmpeg)
assembly stream                     # from the microphone; Ctrl-C to stop
assembly stream --system-audio      # macOS: system/app audio + mic as separate sessions
assembly stream --system-audio-only # macOS: system/app audio without the mic
```

`assembly stream` exposes the full `StreamingParameters` surface (model & input, turn detection, features) as curated flags, with the same `--config` / `--config-file` escape hatch:

```sh
assembly stream --sample --max-turn-silence 400 --format-turns \
  --keyterms-prompt "AssemblyAI" --config vad_threshold=0.7
```

On macOS, `--system-audio` uses ScreenCaptureKit to capture system/app audio without a loopback driver and labels finalized turns `You:` or `System:`. The first run may prompt for Screen & System Audio Recording and Microphone permissions.

## Live Transcript → Live LLM

Run a prompt over the live transcript through the LLM Gateway, refreshing on every finalized turn — one command, no pipe to wire up:

```sh
assembly stream --llm "summarize action items as I talk"
assembly stream --llm "extract action items" --llm "rewrite them as a checklist"  # chains
```

On a terminal you watch one evolving panel; add `--json` for one JSON object per refresh. Prefer the pipe? Compose the primitives — `assembly stream -o text` writes one finalized turn per line and `assembly llm -f` re-runs your prompt over the growing transcript:

```sh
assembly stream -o text | assembly llm -f --system "You are a meeting scribe" "summarize action items"
```

## Voice Agent

Have a live, two-way voice conversation — full-duplex, so you can interrupt mid-sentence (barge-in). **Use headphones**, otherwise the agent hears itself. (This only *runs* a conversation; to **build** a voice-agent app, use `assembly init voice-agent`.)

```sh
assembly agent                                    # talk; the agent talks back. Ctrl-C to stop.
assembly agent --voice james --greeting "Hi"
assembly agent --system-prompt-file persona.txt   # load the system prompt from a file
assembly agent --list-voices                      # see available voices
```

## Show The Code

Add `--show-code` to `transcribe`, `stream`, or `agent` to print the equivalent Python SDK script **instead of running** — a ready-to-edit starting point built from exactly the flags you passed. It needs no API key (generated code reads `ASSEMBLYAI_API_KEY`) and writes plain Python to stdout:

```sh
assembly transcribe --sample --speaker-labels --show-code        # print the equivalent script
assembly transcribe call.mp3 --sentiment-analysis --show-code > my_transcribe.py
assembly stream --show-code                                      # the microphone-streaming idiom
assembly agent --voice ivy --show-code                           # the full-duplex agent loop
```

With `--llm` (repeatable), it emits the chained LLM Gateway calls too.

## Pipelines

`assembly` composes with the rest of your shell. Output is machine-clean (errors → stderr), commands read `-` from stdin, and `-o`/`--output` prints a single field so you rarely need `jq`.

```sh
# Pick one field with -o
assembly transcribe call.mp3 -o text         # just the transcript text
assembly transcribe video.mp4 -o srt         # SubRip (.srt) captions
assembly transcribe call.mp3 --json | jq .   # full JSON when you do want jq

# Read audio from stdin
ffmpeg -i talk.mp4 -f wav - | assembly transcribe -          # transcribe any video
curl -sL https://example.com/ep.mp3 | assembly transcribe -  # no temp file

# assembly llm is a general text filter — it reads stdin, audio optional
git log --oneline -30 | assembly llm "write release notes grouped by feature/fix"

# DIY voice assistant — speak a question, hear the answer (use headphones)
assembly stream -o text | while IFS= read -r line; do
  echo "$line" | assembly llm -o text "answer in one short sentence" | say
done
```

A Ctrl-C in a pipe hits both sides; to stop just the producer and let the consumer finish, signal the producer (`timeout -s INT 30s assembly stream …`) or end on a natural pause (`assembly stream --inactivity-timeout 5`).

## API Key & Security

`assembly` resolves your key in order: the `ASSEMBLYAI_API_KEY` environment variable, then the OS keyring (written only by `assembly login`). Two things worth knowing:

- The key is **never stored in a plaintext dotfile** — `assembly login` puts it in the OS keyring (Keychain / Credential Manager / Secret Service); the only on-disk config holds just profile names.
- There is **no `--api-key` flag on run commands**, so a key can't leak into `ps` output or shell history.

Prefer not to persist it? Set the env var instead — it's checked *before* the keyring, so nothing is written to disk. Scope it to one command (and keep it out of history) by injecting from a secret manager at call time:

```sh
ASSEMBLYAI_API_KEY=$(op read "op://Private/AssemblyAI/api key") assembly transcribe call.mp3
op run -- assembly transcribe call.mp3                    # …or wrap the whole command
```

In CI, set `ASSEMBLYAI_API_KEY` as a masked secret. `assembly logout` purges the keyring entry; `assembly whoami` / `assembly doctor` confirm the active source without printing the key.

## Telemetry

`assembly` collects **anonymous** usage telemetry to help improve the CLI: the command name (never its arguments), outcome class and exit code, duration, CLI version, OS, Python version, whether it ran in CI, and a random install id. It never collects arguments, file paths or contents, transcripts, API keys, or account data — and delivery runs in a detached background process, so it never slows a command down.

Opt out any time, persistently or per-environment:

```sh
assembly telemetry disable     # persisted on this machine (assembly telemetry status to inspect)
export AAI_TELEMETRY_DISABLED=1   # env kill-switch; the cross-tool DO_NOT_TRACK=1 also works
```

The ingestion credential in the source is a Datadog **client token** — the write-only, embeddable credential class (it can submit events, read nothing). No account secret ships with the CLI.

## Account Self-Service

These commands use your browser login session (run `assembly login`), not your API key:

```sh
assembly keys list                       # list API keys (masked) across projects
assembly keys create --name ci-pipeline  # mint a new key (printed once)
assembly balance                         # remaining account balance
assembly usage --start 2026-05-01 --end 2026-06-01
assembly sessions list --status completed
assembly audit --action token.create     # account audit log, filterable
```

AMS sessions are short-lived — if a command reports it needs a browser login, run `assembly login` again.

## AI Coding Agents

Set your coding agent up for AssemblyAI — the live docs (MCP server), the AssemblyAI skill, and the bundled `aai-cli` skill — so your agent writes current, correct integration code:

```sh
assembly setup install        # docs MCP + assemblyai skill + bundled aai-cli skill (user scope)
assembly setup status         # show what's set up
assembly setup remove         # unwind all three
```

`install` shells out to `claude mcp add` for the MCP and `npx skills add` for the `assemblyai` skill; the `aai-cli` skill ships inside the package and is copied in directly (no network). Pass `--scope project` to scope the MCP server to the current project. A missing `claude` or `npx` is reported and skipped, not treated as an error.

## Reference

Use `--help` on any command to explore flags and examples:

```sh
assembly --help
assembly transcribe --help
assembly stream --help
```

- [AssemblyAI docs](https://www.assemblyai.com/docs)
- [API reference](https://www.assemblyai.com/docs/api-reference)

## Development

This project uses [uv](https://docs.astral.sh/uv/). Run tools through `uv run` so they use the locked environment (`pyproject.toml` + `uv.lock`):

```sh
uv sync --extra dev        # create/refresh the venv with dev dependencies
uv run assembly --help          # run the CLI from the locked environment
uv run pytest              # run the test suite (uv run mypy / ruff likewise)
./scripts/check.sh         # ruff + mypy + pytest — the same checks CI runs on every PR
```

## License

Released under the [MIT license](LICENSE).
</content>
</invoke>
