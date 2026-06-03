# AssemblyAI CLI (`aai`)

A command-line interface for [AssemblyAI](https://www.assemblyai.com): transcribe
files, stream live audio, and have two-way voice conversations — all from your terminal.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh
```

The installer uses [`pipx`](https://pipx.pypa.io) when available (falling back to
`pip --user`) and requires Python 3.10+. Prefer to do it yourself:

```sh
pipx install "git+https://github.com/AssemblyAI/cli.git"   # or: pip install --user ...
```

Microphone and speaker support (for `stream` and `agent`) is **included by default** —
no extra install step. Audio runs on [`sounddevice`](https://python-sounddevice.readthedocs.io),
whose macOS and Windows wheels bundle PortAudio, so there's nothing else to install. On Linux,
install the PortAudio runtime once (`sudo apt-get install libportaudio2`).

## Quick start

```sh
aai login                 # store your API key (browser-assisted)
aai transcribe --sample   # transcribe the hosted wildfires.mp3 sample
```

## Commands

| Command | What it does |
| --- | --- |
| `aai login` / `logout` / `whoami` | Manage the stored API key. |
| `aai transcribe <file\|url>` | Transcribe an audio file or URL (`--sample` for a demo, `--srt`/`--vtt` for subtitles). |
| `aai transcripts list` / `get <id>` | Browse and fetch past transcripts. |
| `aai stream [file]` | Real-time transcription from a file or the microphone. |
| `aai agent` | Live two-way voice conversation with a voice agent. |
| `aai claude install` | Wire Claude Code up to AssemblyAI's docs + skill. |
| `aai samples create <name>` | Scaffold a runnable starter script with your key injected. |

Add `--json` to any command for machine-readable output (it's also the default when
output is piped or run by an agent). Auth problems surface as a clean
"not authenticated" error across every command.

## Streaming

```sh
aai stream --sample            # stream the hosted wildfires.mp3 sample (same clip as transcribe)
aai stream path/to/audio.wav   # 16 kHz mono WAV streams directly
aai stream path/to/audio.mp3   # other formats need ffmpeg on PATH
aai stream https://…/clip.mp3  # a URL works too (decoded via ffmpeg)
aai stream                     # from the microphone; Ctrl-C to stop
```

## Voice agent

Have a live, two-way voice conversation:

```sh
aai agent                                 # talk; the agent talks back. Ctrl-C to stop.
aai agent --voice james --greeting "Hi"
aai agent --prompt-file persona.txt       # load the system prompt from a file
aai agent --list-voices                   # see available voices
```

By default the agent runs **half-duplex**: your mic mutes while the agent speaks, so it
can't hear itself on your speakers. With headphones, add `--full-duplex` for true
barge-in (interrupt the agent mid-sentence).

## AI coding agents

Wire Claude Code up to AssemblyAI's live docs (MCP server) and the AssemblyAI skill so
your agent writes current, correct integration code:

```sh
aai claude install        # installs the docs MCP server + skill (user scope)
aai claude status         # show what's wired up
aai claude remove         # unwind both
```

`install` shells out to `claude mcp add` and `npx skills add`. Pass `--scope project` to
scope the MCP server to the current project. A missing `claude` or `npx` is reported and
skipped (with the manual command to run), not treated as an error.

## Development

```sh
pip install -e ".[dev]"
./scripts/check.sh         # ruff + mypy + pytest (the same checks CI runs on every PR)
```
