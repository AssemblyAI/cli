# AssemblyAI CLI

[![Python](https://img.shields.io/badge/python-3.12+-D6402E)](https://github.com/AssemblyAI/cli)
[![License](https://img.shields.io/badge/license-MIT-D6402E)](https://github.com/AssemblyAI/cli/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-assemblyai-D6402E)](https://www.assemblyai.com/docs)

The AssemblyAI CLI (`assembly`) brings speech AI to your terminal: transcribe files, URLs, and YouTube/podcast pages, stream live audio, talk to a two-way voice agent, prompt the LLM Gateway, benchmark speech models, and scaffold ready-to-deploy starter apps.

## 🚀 Why the AssemblyAI CLI?

- **🎯 One command for everything**: transcription, real-time streaming, voice agents, LLM prompts, and WER benchmarking — no SDK boilerplate.
- **🔌 Built for pipelines**: data goes to stdout, errors to stderr, `--json` gives stable machine-readable output, and `-` reads audio from stdin.
- **🔐 Secure by default**: your API key lives in the OS keyring, never in a dotfile — and run commands have no `--api-key` flag, so keys can't leak into `ps` or shell history.
- **🛠️ From demo to deployed app**: `assembly init` scaffolds a runnable FastAPI starter, `assembly dev` / `share` / `deploy` run, tunnel, and ship it, and `--show-code` prints the equivalent Python SDK script for any run command.
- **🤖 Agent-ready**: `assembly setup install` wires your coding agent up with the AssemblyAI docs MCP server and skills.
- **📖 Open source**: MIT licensed.

## 📦 Installation

Requires Python 3.12+ (Homebrew brings its own; for pipx/uv see the `--python` hint below).

> ⚠️ The `assemblyai-cli` package on PyPI is **not** this project — install with one of the
> commands below, not `pip install assemblyai-cli`.

### Homebrew (recommended — macOS / Linux)

```sh
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew trust assemblyai/cli   # only needed when HOMEBREW_REQUIRE_TAP_TRUST is set; harmless otherwise
brew install assembly
```

Homebrew pulls in `ffmpeg` and `portaudio`, so every command works out of the box.

### pipx / uv

```sh
pipx install "git+https://github.com/AssemblyAI/cli.git"
# or
uv tool install "git+https://github.com/AssemblyAI/cli.git"
```

If your default interpreter is older than Python 3.12, add `--python python3.12` (pipx) or
`--python 3.12` (uv) to the install command.

Only the live-audio commands need anything extra: `stream` and `agent` use PortAudio for
microphone capture (Debian/Ubuntu: `sudo apt-get install libportaudio2`; Fedora:
`sudo dnf install portaudio`) and [`ffmpeg`](https://ffmpeg.org) on `PATH` to stream
non-WAV audio. Plain `transcribe` uploads your file directly and needs neither.

## 🔐 Authentication

New to AssemblyAI? Create a free account at
[assemblyai.com/dashboard](https://www.assemblyai.com/dashboard) to get an API key.

The easiest path is browser login, which stores your API key in the OS keyring
(Keychain / Credential Manager / Secret Service):

```sh
assembly login
```

In CI — or anywhere a browser isn't an option — set the key as an environment variable
instead. It's checked before the keyring, and nothing is written to disk:

```sh
export ASSEMBLYAI_API_KEY="YOUR_API_KEY"
```

## 🚀 Getting started

For a guided tour — sign in, run a first transcription, start building:

```sh
assembly onboard
```

Or jump straight in:

```sh
assembly transcribe --sample   # transcribe the hosted sample file
assembly transcribe call.mp3   # then your own audio
assembly stream --sample       # live streaming, no microphone needed
assembly stream                # stream your microphone (Ctrl-C to stop)
assembly agent                 # talk to a voice agent (use headphones)
assembly init                  # scaffold a starter app
```

## 📋 Key features

- **Transcription**: `assembly transcribe` handles files, URLs, and YouTube/podcast pages, with flags for speaker labels, PII redaction, summarization, sentiment, chapters, and more.
- **Batch transcription**: point `assembly transcribe` at a directory or glob (or pipe paths with `--from-stdin`) to transcribe everything concurrently, with sidecar files that make re-runs resumable. Add `--llm "prompt"` to run an LLM prompt over each finished transcript, saved into the sidecars.
- **Real-time streaming**: `assembly stream` transcribes the microphone, a file, or a URL live — on macOS it can capture system audio too.
- **Voice agent**: `assembly agent` runs a full-duplex spoken conversation in your terminal.
- **LLM Gateway**: `assembly llm` prompts an LLM over a transcript, stdin, or a live stream (`assembly stream --llm "summarize as I talk"`).
- **Model evaluation**: `assembly eval` transcribes a Hugging Face dataset (with built-in aliases for common benchmarks: `assembly eval tedlium`) or a local `.csv`/`.jsonl` manifest and scores WER against its references — handy for picking a speech model.
- **Starter apps**: `assembly init` scaffolds a self-contained FastAPI + HTML app (`audio-transcription`, `live-captions`, `voice-agent`); `assembly dev` runs it, `assembly share` exposes it on a public URL, and `assembly deploy` ships it to Vercel, Railway, or Fly.io.
- **Webhook testing**: `assembly webhooks listen` opens a public dev URL (cloudflared quick tunnel) that prints webhook deliveries as they arrive and can forward them to your local app with `--forward-to`.
- **Code generation**: add `--show-code` to `transcribe`/`stream`/`agent` to print the equivalent Python SDK script instead of running.
- **Account self-service**: `assembly keys` / `balance` / `usage` / `limits` / `sessions` / `audit` via browser login.

### Quick examples

Pull exactly the output you need:

```sh
assembly transcribe call.mp3 -o text   # just the text
assembly transcribe video.mp4 -o srt   # captions
assembly transcribe call.mp3 --speaker-labels --summarization --json
```

Transcribe in batches — a directory, a glob, or a piped list, resumable on re-run:

```sh
assembly transcribe ./recordings
find . -name "*.wav" | assembly transcribe --from-stdin
```

Compose with other tools — audio in, text out:

```sh
ffmpeg -i talk.mp4 -f wav - | assembly transcribe -
git log --oneline -30 | assembly llm "write release notes grouped by feature/fix"
```

Graduate to the SDK — print the equivalent Python script instead of running:

```sh
assembly transcribe --sample --speaker-labels --show-code
```

## 📚 Documentation

- Run `assembly --help` or `assembly <command> --help` for flags and examples.
- Run `assembly doctor` to check your environment (API key, network, ffmpeg, microphone).
- [AssemblyAI docs](https://www.assemblyai.com/docs)
- [API reference](https://www.assemblyai.com/docs/api-reference)

## 🤝 Contributing

This project uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync                  # create/refresh the venv
uv run assembly --help   # run the CLI from the locked environment
./scripts/check.sh       # the full gate CI runs
```

See [AGENTS.md](AGENTS.md) for development conventions and architecture notes.

## 📄 Legal

Released under the [MIT license](LICENSE).
