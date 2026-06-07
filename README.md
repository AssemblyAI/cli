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
  <a href="https://github.com/AssemblyAI/cli/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/AssemblyAI/cli?style=flat-square&color=D6402E"></a>
  <a href="https://www.assemblyai.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-assemblyai-D6402E?style=flat-square"></a>
</p>

---

`aai` brings [AssemblyAI](https://www.assemblyai.com) to your terminal: transcribe files, stream live audio, run a two-way voice agent, prompt the LLM Gateway, and scaffold ready-to-deploy starter apps — all pipeline-friendly, with your key kept in the OS keyring.

## Installation

```sh
# YOLO
curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh

# pipx (recommended)
pipx install "git+https://github.com/AssemblyAI/cli.git"
```

Requires Python 3.12+. The installer prefers [`pipx`](https://pipx.pypa.io), falling back to `pip --user`. Microphone and speaker support (for `stream` and `agent`) is included by default via [`sounddevice`](https://python-sounddevice.readthedocs.io) — its macOS and Windows wheels bundle PortAudio. On Linux, install the runtime once: `sudo apt-get install libportaudio2`.

### Homebrew (macOS / Linux)

```sh
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew install aai
```

`brew install` pulls in `ffmpeg` and `portaudio` for you, so `transcribe`, `stream`, and `agent` work out of the box. Upgrade with `brew upgrade aai`; remove with `brew uninstall aai`.

## Quick Start

```sh
aai login                 # store your API key (browser-assisted)
aai transcribe --sample   # transcribe the hosted wildfires.mp3 sample
```

## Build An App

`aai init` is how you **build a new app** — it copies a small, self-contained FastAPI + HTML project you can run locally and deploy to Vercel as-is. This is the starting point whenever you want to *create* something, including a voice agent app:

```sh
aai init                            # pick a template, scaffold, install deps, open the browser
aai init audio-transcription myapp  # non-interactive: template + directory
aai init voice-agent my-agent       # build a voice agent app (full FastAPI + browser starter)
```

Your key is written to a git-ignored `.env` (never sent to the browser). Use `--no-install` to scaffold only.

> **Building a voice agent? Use `aai init voice-agent`, not `aai agent`.** `aai agent` only *runs* a live mic conversation in the terminal and writes no code; `aai init` creates the actual app.

## Commands

| Command | What it does |
| --- | --- |
| `aai login` / `logout` / `whoami` | Manage the stored API key. |
| `aai doctor` | Check your environment (API key, network, ffmpeg, microphone, agent tooling). |
| `aai transcribe <file\|url>` | Transcribe a file, URL, or YouTube URL (`--sample`, `--llm`, `--show-code`). |
| `aai transcripts list` / `get <id>` | Browse and fetch past transcripts. |
| `aai stream [file]` | Real-time transcription from a file or the microphone. |
| `aai agent` | *Run* a live two-way voice conversation (to **build** a voice agent app, use `aai init voice-agent`). |
| `aai llm <prompt>` | Prompt the LLM Gateway (`--transcript-id`, or `--follow` for a live stream). |
| `aai setup install` | Set up your coding agent for AssemblyAI (docs MCP + skills). |
| `aai samples create <name>` | Scaffold a runnable starter script. |
| `aai keys` / `balance` / `usage` / `limits` / `sessions` / `audit` | Account self-service (browser login). |

Every command prints human-readable text by default — in a terminal, a pipe, CI, or under an agent alike. Add `--json` for machine-readable output (it never switches on you just because stdout is piped, so `aai transcribe call.mp3 | grep hello` still gets the transcript, not a JSON blob). Errors go to **stderr**, so stdout stays clean for pipelines.

> **Tip:** Quote URLs that contain `?` (most YouTube links do) — in zsh the `?` is a glob character: `aai transcribe "https://www.youtube.com/watch?v=VIDEO_ID"`.

## Transcribe A File

`aai transcribe` exposes the full `TranscriptionConfig` surface as curated, grouped flags — model & language, formatting, speakers & channels, PII/safety guardrails, analysis (summary, chapters, sentiment, entities, topics, highlights), customization, and webhooks:

```sh
aai transcribe call.mp3 \
  --speaker-labels --speakers-expected 2 \
  --redact-pii --redact-pii-policy person_name,phone_number \
  --summarization --summary-type bullets \
  --sentiment-analysis --auto-chapters \
  --config speech_threshold=0.5 \
  --config-file extra.json
```

Anything without a curated flag is reachable via the escape hatch: `--config KEY=VALUE` (repeatable) and `--config-file FILE` (a JSON object) accept any SDK field by name. Precedence: config file < `--config` < explicit flags. Run `aai transcribe --help` for the full flag list.

## Stream Live Audio

```sh
aai stream --sample            # stream the hosted wildfires.mp3 sample
aai stream path/to/audio.wav   # 16 kHz mono WAV streams directly (other formats need ffmpeg)
aai stream https://…/clip.mp3  # a URL works too (decoded via ffmpeg)
aai stream                     # from the microphone; Ctrl-C to stop
aai stream --system-audio      # macOS: system/app audio + mic as separate sessions
aai stream --system-audio-only # macOS: system/app audio without the mic
```

`aai stream` exposes the full `StreamingParameters` surface (model & input, turn detection, features) as curated flags, with the same `--config` / `--config-file` escape hatch:

```sh
aai stream --sample --max-turn-silence 400 --format-turns \
  --keyterms-prompt "AssemblyAI" --config vad_threshold=0.7
```

On macOS, `--system-audio` uses ScreenCaptureKit to capture system/app audio without a loopback driver and labels finalized turns `You:` or `System:`. The first run may prompt for Screen & System Audio Recording and Microphone permissions.

## Live Transcript → Live LLM

Run a prompt over the live transcript through the LLM Gateway, refreshing on every finalized turn — one command, no pipe to wire up:

```sh
aai stream --llm "summarize action items as I talk"
aai stream --llm "extract action items" --llm "rewrite them as a checklist"  # chains
```

On a terminal you watch one evolving panel; add `--json` for one JSON object per refresh. Prefer the pipe? Compose the primitives — `aai stream -o text` writes one finalized turn per line and `aai llm -f` re-runs your prompt over the growing transcript:

```sh
aai stream -o text | aai llm -f --system "You are a meeting scribe" "summarize action items"
```

## Voice Agent

Have a live, two-way voice conversation — full-duplex, so you can interrupt mid-sentence (barge-in). **Use headphones**, otherwise the agent hears itself. (To **build** a voice agent *app*, use `aai init voice-agent` instead — this command just runs a conversation in the terminal.)

```sh
aai agent                                    # talk; the agent talks back. Ctrl-C to stop.
aai agent --voice james --greeting "Hi"
aai agent --system-prompt-file persona.txt   # load the system prompt from a file
aai agent --list-voices                      # see available voices
```

## Show The Code

Add `--show-code` to `transcribe`, `stream`, or `agent` to print the equivalent Python SDK script **instead of running** — a ready-to-edit starting point built from exactly the flags you passed. It needs no API key (generated code reads `ASSEMBLYAI_API_KEY`) and writes plain Python to stdout:

```sh
aai transcribe --sample --speaker-labels --show-code        # print the equivalent script
aai transcribe call.mp3 --sentiment-analysis --show-code > my_transcribe.py
aai stream --show-code                                      # the microphone-streaming idiom
aai agent --voice ivy --show-code                           # the full-duplex agent loop
```

With `--llm` (repeatable), it emits the chained LLM Gateway calls too.

## Pipelines

`aai` composes with the rest of your shell. Output is machine-clean (errors → stderr), commands read `-` from stdin, and `-o`/`--output` prints a single field so you rarely need `jq`.

```sh
# Pick one field with -o
aai transcribe call.mp3 -o text         # just the transcript text
aai transcribe video.mp4 -o srt         # SubRip (.srt) captions
aai transcribe call.mp3 --json | jq .   # full JSON when you do want jq

# Read audio from stdin
ffmpeg -i talk.mp4 -f wav - | aai transcribe -          # transcribe any video
curl -sL https://example.com/ep.mp3 | aai transcribe -  # no temp file

# aai llm is a general text filter — it reads stdin, audio optional
git log --oneline -30 | aai llm "write release notes grouped by feature/fix"

# DIY voice assistant — speak a question, hear the answer (use headphones)
aai stream -o text | while IFS= read -r line; do
  echo "$line" | aai llm -o text "answer in one short sentence" | say
done
```

A Ctrl-C in a pipe hits both sides; to stop just the producer and let the consumer finish, signal the producer (`timeout -s INT 30s aai stream …`) or end on a natural pause (`aai stream --inactivity-timeout 5`).

## API Key & Security

`aai` resolves your key in order: the `ASSEMBLYAI_API_KEY` environment variable, then the OS keyring (written only by `aai login`). Two things worth knowing:

- The key is **never stored in a plaintext dotfile** — `aai login` puts it in the OS keyring (Keychain / Credential Manager / Secret Service); the only on-disk config holds just profile names.
- There is **no `--api-key` flag on run commands**, so a key can't leak into `ps` output or shell history.

Prefer not to persist it? Set the env var instead — it's checked *before* the keyring, so nothing is written to disk. Scope it to one command (and keep it out of history) by injecting from a secret manager at call time:

```sh
ASSEMBLYAI_API_KEY=$(op read "op://Private/AssemblyAI/api key") aai transcribe call.mp3
op run -- aai transcribe call.mp3                    # …or wrap the whole command
```

In CI, set `ASSEMBLYAI_API_KEY` as a masked secret. `aai logout` purges the keyring entry; `aai whoami` / `aai doctor` confirm the active source without printing the key.

## Account Self-Service

These commands use your browser login session (run `aai login`), not your API key:

```sh
aai keys list                       # list API keys (masked) across projects
aai keys create --name ci-pipeline  # mint a new key (printed once)
aai balance                         # remaining account balance
aai usage --start 2026-05-01 --end 2026-06-01
aai sessions list --status completed
aai audit --action token.create     # account audit log, filterable
```

AMS sessions are short-lived — if a command reports it needs a browser login, run `aai login` again.

## AI Coding Agents

Set your coding agent up for AssemblyAI — the live docs (MCP server), the AssemblyAI skill, and the bundled `aai-cli` skill — so your agent writes current, correct integration code:

```sh
aai setup install        # docs MCP + assemblyai skill + bundled aai-cli skill (user scope)
aai setup status         # show what's set up
aai setup remove         # unwind all three
```

`install` shells out to `claude mcp add` for the MCP and `npx skills add` for the `assemblyai` skill; the `aai-cli` skill ships inside the package and is copied in directly (no network). Pass `--scope project` to scope the MCP server to the current project. A missing `claude` or `npx` is reported and skipped, not treated as an error.

## Reference

Use `--help` on any command to explore flags and examples:

```sh
aai --help
aai transcribe --help
aai stream --help
```

- [AssemblyAI docs](https://www.assemblyai.com/docs)
- [API reference](https://www.assemblyai.com/docs/api-reference)

## Development

This project uses [uv](https://docs.astral.sh/uv/). Run tools through `uv run` so they use the locked environment (`pyproject.toml` + `uv.lock`):

```sh
uv sync --extra dev        # create/refresh the venv with dev dependencies
uv run aai --help          # run the CLI from the locked environment
uv run pytest              # run the test suite (uv run mypy / ruff likewise)
./scripts/check.sh         # ruff + mypy + pytest — the same checks CI runs on every PR
```

## License

Released under the [MIT license](LICENSE).
</content>
</invoke>
