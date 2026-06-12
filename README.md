# AssemblyAI CLI

[![Python](https://img.shields.io/badge/python-3.12+-D6402E)](https://github.com/AssemblyAI/cli)
[![License](https://img.shields.io/badge/license-MIT-D6402E)](https://github.com/AssemblyAI/cli/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-assemblyai-D6402E)](https://www.assemblyai.com/docs)

The AssemblyAI CLI (`assembly`) brings speech AI to your terminal: transcribe files, URLs, and YouTube/podcast pages, stream live audio, talk to a two-way voice agent, prompt the LLM Gateway, benchmark speech models, and scaffold ready-to-deploy starter apps.

<p align="center">
  <img src="assets/welcome.png" alt="The assembly CLI welcome screen, listing command groups for transcription, streaming, voice agents, app scaffolding, and account management" width="820">
</p>

## 🚀 Why the AssemblyAI CLI?

- **🎯 One command for everything**: transcription, real-time streaming, voice agents, LLM prompts, and WER benchmarking — no SDK boilerplate.
- **🔌 Built for pipelines**: data goes to stdout, errors to stderr, `--json` gives stable machine-readable output, and `-` reads audio from stdin.
- **🔐 Secure by default**: your API key lives in the OS keyring, never in a dotfile — and run commands have no `--api-key` flag, so keys can't leak into `ps` or shell history.
- **🛠️ From demo to deployed app**: `assembly init` scaffolds a runnable FastAPI starter, `assembly dev` / `share` / `deploy` run, tunnel, and ship it, and `--show-code` prints the equivalent Python SDK script for any run command.
- **🤖 Agent-ready**: `assembly setup install` wires your coding agent up with the AssemblyAI docs MCP server and skills.
- **📖 Open source**: MIT licensed.

## ✨ Things you can do with it

A few one-liners that show what `assembly` can do. These are the fun ones; the everyday basics live under **Quick examples** below.

**Recreate a scene with synthetic voices** — transcribe and diarize a YouTube clip, then pipe it straight into TTS with a different voice per speaker:

```sh
assembly transcribe "https://www.youtube.com/watch?v=awmCtXzFsJo" --speaker-labels \
  | assembly --sandbox speak --voice A=jane --voice B=mary --out scene.wav
```

`speak` auto-detects `Speaker A:` labels, merges each speaker's turns, and rotates voices. (`speak` is sandbox-only today, hence `--sandbox`.)

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

Only the live-audio commands need anything extra: `stream`, `dictate`, and `agent` use PortAudio for
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
- **Batch transcription**: point `assembly transcribe` at a directory or glob — local, or in bucket storage (`"s3://bucket/calls/*.mp3"`, `gs://`, `az://`, …, with the matching fsspec backend such as `s3fs` installed) — or pipe paths with `--from-stdin` to transcribe everything concurrently, with sidecar files that make re-runs resumable. Add `--llm "prompt"` to run an LLM prompt over each finished transcript, saved into the sidecars.
- **Real-time streaming**: `assembly stream` transcribes the microphone, a file, or a URL live — on macOS it can capture system audio too.
- **Dictation**: `assembly dictate` is push-to-talk for your terminal — press Enter to record, Enter again to get the utterance back instantly from the Sync API (up to 120 s per utterance).
- **Voice agent**: `assembly agent` runs a full-duplex spoken conversation in your terminal.
- **LLM Gateway**: `assembly llm` prompts an LLM over a transcript, stdin, or a live stream (`assembly stream --llm "summarize as I talk"`).
- **Transcript-driven clipping**: `assembly clip` cuts an audio/video file (or a YouTube/podcast URL) with ffmpeg by diarized speaker (`--speaker A`), text match (`--search "pricing"`), LLM pick (`--llm "the three best moments"`), or explicit time range (`--range 1:30-2:45`) — transcribing on the fly, reusing a finished transcript with `-t ID`, or reading one from a pipe (`assembly transcribe x.mp4 --speaker-labels --json | assembly clip x.mp4 -t - --llm "…"`). Clip boundaries snap into nearby silence (ffmpeg `silencedetect`) so cuts don't land mid-word; `--no-snap` cuts at the exact selected times.
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
