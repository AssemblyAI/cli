# AssemblyAI CLI

[![Python](https://img.shields.io/badge/python-3.12+-D6402E)](https://github.com/AssemblyAI/cli)
[![License](https://img.shields.io/badge/license-MIT-D6402E)](https://github.com/AssemblyAI/cli/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-assemblyai-D6402E)](https://www.assemblyai.com/docs)

The AssemblyAI CLI (`assembly`) brings speech AI to your terminal: transcribe files, stream live audio, run a two-way voice agent, prompt the LLM Gateway, and scaffold ready-to-deploy starter apps.

## 🚀 Why the AssemblyAI CLI?

- **🎯 Everything in one command**: transcription, real-time streaming, voice agents, and LLM prompts — no SDK boilerplate.
- **🔌 Pipeline-friendly**: data goes to stdout, errors to stderr, `--json` for stable machine-readable output, `-` reads audio from stdin.
- **🔐 Secure by default**: your API key lives in the OS keyring, never in a dotfile, and run commands have no `--api-key` flag so keys can't leak into shell history.
- **🛠️ From demo to app**: `assembly init` scaffolds a runnable FastAPI starter app, and `--show-code` prints the equivalent Python SDK script for any command.
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

Homebrew pulls in `ffmpeg` and `portaudio`, so `stream` and `agent` work out of the box.
Plain `transcribe` uploads your file directly and needs neither.

### pipx / uv

With pipx:

```sh
pipx install "git+https://github.com/AssemblyAI/cli.git"
```

Or with uv:

```sh
uv tool install "git+https://github.com/AssemblyAI/cli.git"
```

If your default interpreter is older than Python 3.12, add `--python python3.12` (pipx) or
`--python 3.12` (uv) to the install command.

Only `stream` and `agent` need extras: on Linux, install PortAudio once for microphone support
(Debian/Ubuntu: `sudo apt-get install libportaudio2`; Fedora: `sudo dnf install portaudio`), and
have [`ffmpeg`](https://ffmpeg.org) on `PATH` to stream non-WAV audio. Plain `transcribe` needs
neither.

## 📋 Key Features

- **Transcription**: `assembly transcribe` handles files, URLs, and YouTube/podcast pages, with flags for speaker labels, PII redaction, summarization, sentiment, chapters, and more.
- **Batch transcription**: point `assembly transcribe` at a directory or glob (or pipe paths with `--from-stdin`) to transcribe everything concurrently, with sidecar files that make re-runs resumable.
- **Real-time streaming**: `assembly stream` transcribes the microphone, a file, or a URL live — on macOS it can capture system audio too.
- **Voice agent**: `assembly agent` runs a full-duplex spoken conversation in your terminal (use headphones).
- **LLM Gateway**: `assembly llm` prompts an LLM over a transcript, stdin, or a live stream (`assembly stream --llm "summarize as I talk"`).
- **Model evaluation**: `assembly eval` transcribes a Hugging Face dataset or a local `.csv`/`.jsonl` manifest and scores WER against its references (plus DER with `--speaker-labels`) — handy for picking a speech model.
- **Starter apps**: `assembly init` scaffolds a self-contained FastAPI + HTML app (`audio-transcription`, `live-captions`, `voice-agent`).
- **Code generation**: add `--show-code` to `transcribe`/`stream`/`agent` to print the equivalent Python SDK script instead of running.
- **Account self-service**: `assembly keys` / `balance` / `usage` / `limits` / `sessions` / `audit` via browser login.

## 🔐 Authentication

New to AssemblyAI? Create a free account at
[assemblyai.com/dashboard](https://www.assemblyai.com/dashboard) to get an API key.

### Option 1: Browser login (recommended)

```sh
assembly login
```

Stores your API key in the OS keyring (Keychain / Credential Manager / Secret Service).

### Option 2: Environment variable

```sh
export ASSEMBLYAI_API_KEY="YOUR_API_KEY"
```

Checked before the keyring, so nothing is written to disk — ideal for CI (set it as a masked secret).

## 🚀 Getting Started

### Basic usage

Guided setup: sign in, first transcription, start building:

```sh
assembly onboard
```

Transcribe the hosted sample:

```sh
assembly transcribe --sample
```

Then your own audio:

```sh
assembly transcribe call.mp3
```

Stream the hosted sample live (no microphone needed):

```sh
assembly stream --sample
```

Or stream your microphone (Ctrl-C to stop):

```sh
assembly stream
```

Talk to a voice agent:

```sh
assembly agent
```

Scaffold a starter app:

```sh
assembly init
```

### Quick examples

Just the text:

```sh
assembly transcribe call.mp3 -o text
```

Or captions:

```sh
assembly transcribe video.mp4 -o srt
```

Speaker labels + summary, as JSON:

```sh
assembly transcribe call.mp3 --speaker-labels --summarization --json
```

Batch: a whole directory or glob, resumable on re-run:

```sh
assembly transcribe ./recordings
```

Or pipe paths in:

```sh
find . -name "*.wav" | assembly transcribe --from-stdin
```

Pipe audio in, pipe text out:

```sh
ffmpeg -i talk.mp4 -f wav - | assembly transcribe -
```

Prompt the LLM Gateway over any text:

```sh
git log --oneline -30 | assembly llm "write release notes grouped by feature/fix"
```

Print the equivalent Python SDK script instead of running:

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

Create/refresh the venv:

```sh
uv sync
```

Run the CLI from the locked environment:

```sh
uv run assembly --help
```

Run the full gate CI runs:

```sh
./scripts/check.sh
```

See [AGENTS.md](AGENTS.md) for development conventions and architecture notes.

## 📄 Legal

Released under the [MIT license](LICENSE).
