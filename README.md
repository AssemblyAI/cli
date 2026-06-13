# AssemblyAI CLI

[![Python](https://img.shields.io/badge/python-3.12+-D6402E)](https://github.com/AssemblyAI/cli)
[![License](https://img.shields.io/badge/license-MIT-D6402E)](https://github.com/AssemblyAI/cli/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-assemblyai-D6402E)](https://www.assemblyai.com/docs)

The AssemblyAI CLI (`assembly`) brings speech AI directly into your terminal: transcribe files, URLs, and YouTube/podcast pages, stream live audio, talk to a two-way voice agent, prompt the LLM Gateway, benchmark speech models, and scaffold ready-to-deploy starter apps.

<p align="center">
  <img src="assets/welcome.png" alt="The assembly CLI welcome screen, listing command groups for transcription, streaming, voice agents, app scaffolding, and account management" width="820">
</p>

Learn more about the platform in the [AssemblyAI docs](https://www.assemblyai.com/docs).

## ⚡ Quickstart

Install on macOS or Linux with Homebrew:

```sh
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew install assembly
```

Sign in (stores your API key in the OS keyring) and run your first transcription:

```sh
assembly login
assembly transcribe --sample
```

That's it. Run `assembly onboard` for a guided tour, or see [Installation](#-installation) for pipx/uv and other options.

## 🚀 Why the AssemblyAI CLI?

- **🎯 One command for everything**: transcription, real-time streaming, voice agents, LLM prompts, and WER benchmarking — no SDK boilerplate.
- **🔌 Built for pipelines**: data goes to stdout, errors to stderr, `--json` gives stable machine-readable output, and `-` reads audio from stdin.
- **🔐 Secure by default**: your API key lives in the OS keyring, never in a dotfile — and run commands have no `--api-key` flag, so keys can't leak into `ps` or shell history.
- **🛠️ From demo to deployed app**: `assembly init` scaffolds a runnable FastAPI starter, `assembly dev` / `share` / `deploy` run, tunnel, and ship it, and `--show-code` prints the equivalent Python SDK script for any run command.
- **🤖 Agent-ready**: `assembly setup install` wires your coding agent up with the AssemblyAI docs MCP server and skills.
- **📖 Open source**: MIT licensed.

## 📋 Features at a glance

| Command | What it does |
| :--- | :--- |
| `assembly transcribe` | Transcribe files, URLs, YouTube/podcast pages, directories, globs, or bucket storage (`s3://`, `gs://`, `az://`) — with speaker labels, PII redaction, summarization, SRT/VTT captions, and resumable batch runs |
| `assembly stream` | Real-time transcription from your microphone, a file, or a URL — on macOS it can capture system audio too |
| `assembly dictate` | Push-to-talk dictation: press Enter to record, Enter again for instant text (Sync STT API, up to 120 s per utterance) |
| `assembly agent` | Full-duplex spoken conversation with a voice agent, right in your terminal |
| `assembly llm` | Prompt the LLM Gateway over a transcript, stdin, or a live stream |
| `assembly clip` | Cut audio/video with ffmpeg by diarized speaker, text match, LLM pick, or time range — clip boundaries snap into nearby silence |
| `assembly dub` | Re-voice an audio/video file in another language: transcription, LLM translation, per-speaker TTS, ffmpeg track-swap (sandbox-only) |
| `assembly speak` | Synthesize text to speech over the streaming-TTS WebSocket (sandbox-only) |
| `assembly eval` | Benchmark WER against Hugging Face datasets (built-in aliases: `librispeech`, `tedlium`, …) or local manifests |
| `assembly init` / `dev` / `share` / `deploy` | Scaffold a FastAPI + HTML starter app, run it locally, expose it on a public URL, ship it to Vercel / Railway / Fly.io |
| `assembly webhooks listen` | Open a public dev URL that prints webhook deliveries and can forward them to your local app |
| `assembly setup` | Wire a coding agent up with the AssemblyAI docs MCP server and skills |
| `assembly keys` / `balance` / `usage` / `limits` / `sessions` / `audit` | Account self-service via browser login |
| `assembly doctor` | Check your environment: API key, network, ffmpeg, microphone |

Add `--show-code` to `transcribe` / `stream` / `agent` to print the equivalent Python SDK script instead of running — the built-in path from CLI experiment to SDK code.

## ✨ Things you can do with it

A few one-liners that show what `assembly` can do. The everyday basics live under [Getting started](#-getting-started) below.

> [!NOTE]
> `speak` and `dub` are sandbox-only today — that's why the examples below pass `--sandbox`.

**Recreate a scene with synthetic voices** — transcribe and diarize a YouTube clip, then pipe it straight into TTS with a different voice per speaker:

```sh
assembly transcribe "https://www.youtube.com/watch?v=awmCtXzFsJo" --speaker-labels \
  | assembly --sandbox speak --voice A=jane --voice B=mary --out scene.wav
```

`speak` auto-detects `Speaker A:` labels, merges each speaker's turns, and rotates voices.

**Dub a video into another language** — the whole platform in one command: transcription with utterance timestamps, per-utterance LLM translation, TTS for each line (one voice per speaker), and ffmpeg laying the new track over the original video:

```sh
assembly --sandbox dub talk.mp4 --lang de
```

The video stream is copied untouched; each dubbed line lands at its original start time.

**Turn a podcast into audio** — Apple and Spotify podcast pages work too (yt-dlp ingestion):

```sh
assembly transcribe "https://podcasts.apple.com/us/podcast/id1516093381" --speaker-labels \
  | assembly --sandbox speak --out episode.wav
```

**Cut the highlight reel from a speech** — `clip` downloads the audio, transcribes it, has an LLM pick the windows, and cuts each one into its own file with ffmpeg (here: Steve Jobs' Stanford commencement address):

```sh
assembly clip "https://www.youtube.com/watch?v=UF8uR6Z6KLc" \
  --llm "the most quotable 20-40 seconds from each of the stories" \
  --padding 0.5 --out-dir .
```

**Burn karaoke subtitles into a music video** — `-o srt` prints captions to stdout, and `--chars-per-caption` keeps the lines short so they flip with the vocals; ffmpeg renders them onto the video (`-f srt -i pipe:` muxes a toggleable soft-subtitle track instead, no re-encode):

```sh
assembly transcribe video.mp4 -o srt --chars-per-caption 24 > lyrics.srt
ffmpeg -i video.mp4 -vf "subtitles=lyrics.srt:force_style='Fontsize=28,PrimaryColour=&H00FFFF&'" karaoke.mp4
```

**Keep a live to-do list from your mic** — `llm -f` re-runs the prompt over the growing transcript, updating in place:

```sh
assembly stream -o text | assembly llm -f "summarize my to-dos as I talk"
```

**Caption a meeting from system audio** (macOS) — captures app/system audio alongside your mic as separate diarized speakers:

```sh
assembly stream --system-audio --speaker-labels -o text
```

**Get pinged when your name comes up** in a live meeting:

```sh
assembly stream -o text | grep --line-buffered -i alex \
  | while read -r _; do afplay /System/Library/Sounds/Glass.aiff; done
```

**Chain LLM prompts over a transcript** — each prompt runs on the finished transcript:

```sh
assembly transcribe --sample --llm "summarize" --llm "translate the summary to French"
```

**Talk to a voice agent in your terminal** — full-duplex, around 20 voices:

```sh
assembly agent --voice ivy --system-prompt "you're a helpful interviewer"
```

**Graduate to the SDK** — `--show-code` prints the equivalent Python script for any `transcribe`/`stream`/`agent` run instead of executing it:

```sh
assembly agent --system-prompt "you're a story generator" --show-code > story.py
```

**Scaffold and deploy a voice agent** — templates: `voice-agent`, `audio-transcription`, `live-captions`:

```sh
assembly init voice-agent && assembly deploy --prod
```

**Benchmark WER against public datasets** — built-in aliases for LibriSpeech, TEDLIUM, and more:

```sh
assembly eval librispeech --speech-model universal-3-pro --limit 50
```

## 📦 Installation

Requires Python 3.12+ (Homebrew brings its own; for pipx/uv see the `--python` hint below).

> [!WARNING]
> The `assemblyai-cli` package on PyPI is **not** this project — install with one of the
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

<details>
<summary>System dependencies for the live-audio commands (pipx/uv installs only)</summary>

Only the live-audio commands need anything extra: `stream`, `dictate`, and `agent` use PortAudio for
microphone capture and [`ffmpeg`](https://ffmpeg.org) on `PATH` to stream non-WAV audio.
Plain `transcribe` uploads your file directly and needs neither.

- Debian/Ubuntu: `sudo apt-get install libportaudio2 ffmpeg`
- Fedora: `sudo dnf install portaudio ffmpeg`
- macOS (Homebrew): `brew install portaudio ffmpeg`

</details>

## 🔐 Authentication

New to AssemblyAI? Create a free account at
[assemblyai.com/dashboard](https://www.assemblyai.com/dashboard) to get an API key.

### Option 1: Browser login (recommended)

**✨ Best for:** day-to-day use on your own machine.

Browser login stores your API key in the OS keyring (Keychain / Credential Manager / Secret Service) — nothing lands in a dotfile, and it unlocks the account commands (`keys`, `balance`, `usage`, `limits`, `sessions`, `audit`):

```sh
assembly login
```

### Option 2: API key environment variable

**✨ Best for:** CI, containers, and anywhere a browser isn't an option.

The environment variable is checked before the keyring, and nothing is written to disk:

```sh
export ASSEMBLYAI_API_KEY="YOUR_API_KEY"
```

## 🚀 Getting started

### Guided tour

Sign in, run a first transcription, start building:

```sh
assembly onboard
```

### Basic usage

```sh
assembly transcribe --sample   # transcribe the hosted sample file
assembly transcribe call.mp3   # then your own audio
assembly stream --sample       # live streaming, no microphone needed
assembly stream                # stream your microphone (Ctrl-C to stop)
assembly agent                 # talk to a voice agent (use headphones)
assembly init                  # scaffold a starter app
```

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
assembly transcribe "s3://bucket/calls/*.mp3"   # needs: pip install s3fs
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

### In the terminal

- Run `assembly --help` or `assembly <command> --help` for flags and examples.
- Run `assembly doctor` to check your environment (API key, network, ffmpeg, microphone).
- Run `assembly onboard` for the guided tour.

### Resources

- [**AssemblyAI docs**](https://www.assemblyai.com/docs) — guides for every model and feature.
- [**API reference**](https://www.assemblyai.com/docs/api-reference) — the REST and streaming APIs the CLI drives.
- [**Dashboard**](https://www.assemblyai.com/dashboard) — manage your account and API keys.
- [**AGENTS.md**](AGENTS.md) — development conventions and architecture notes for contributors.

## 🤝 Contributing

This project uses [uv](https://docs.astral.sh/uv/):

```sh
uv sync                  # create/refresh the venv
uv run assembly --help   # run the CLI from the locked environment
./scripts/check.sh       # the full gate CI runs
```

See [AGENTS.md](AGENTS.md) for development conventions and architecture notes.

## 📄 Legal

- **License**: released under the [MIT license](LICENSE).
- **Privacy**: [AssemblyAI privacy policy](https://www.assemblyai.com/legal/privacy-policy) — the CLI's anonymous usage telemetry is opt-out (`assembly telemetry disable`, `AAI_TELEMETRY_DISABLED=1`, or `DO_NOT_TRACK=1`).
- **Terms**: [AssemblyAI terms of service](https://www.assemblyai.com/legal/terms-of-service).
