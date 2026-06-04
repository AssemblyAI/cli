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
| `aai doctor` | Check your environment is ready (API key, network, ffmpeg, microphone, agent tooling). |
| `aai transcribe <file\|url>` | Transcribe an audio file, URL, or YouTube URL (`--sample` for a demo, `--llm-gateway-prompt` to transform the result, `--show-code` to print the equivalent Python). |
| `aai transcripts list` / `get <id>` | Browse and fetch past transcripts. |
| `aai stream [file]` | Real-time transcription from a file or the microphone. |
| `aai agent` | Live two-way voice conversation with a voice agent. |
| `aai llm <prompt>` | Prompt AssemblyAI's LLM Gateway (optionally over a transcript with `--transcript-id`). |
| `aai claude install` | Wire Claude Code up to AssemblyAI's docs + skill. |
| `aai samples create <name>` | Scaffold a runnable starter script with your key injected. |

Add `--json` to any command for machine-readable output (it's also the default when
output is piped or run by an agent). Auth problems surface as a clean
"not authenticated" error across every command.

> **Tip:** Quote URLs that contain `?` (most YouTube links do). In zsh the `?` is a
> glob character, so an unquoted URL fails with `zsh: no matches found` before the
> command runs:
>
> ```sh
> aai transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
> ```

## Transcribe options

`aai transcribe` exposes the full `TranscriptionConfig` surface as curated flags,
grouped by purpose:

- **Model & language:** `--speech-model`, `--language-code`, `--language-detection`,
  `--keyterms-prompt`, `--prompt`, `--temperature`.
- **Formatting:** `--punctuate` / `--no-punctuate`, `--format-text` /
  `--no-format-text`, `--disfluencies`.
- **Speakers & channels:** `--speaker-labels`, `--speakers-expected`,
  `--multichannel`.
- **Guardrails:** `--redact-pii`, `--redact-pii-policy`, `--redact-pii-sub`,
  `--redact-pii-audio`, `--filter-profanity`, `--content-safety`,
  `--content-safety-confidence`, `--speech-threshold`.
- **Analysis:** `--summarization` (`--summary-type`, `--summary-model`),
  `--auto-chapters`, `--sentiment-analysis`, `--entity-detection`,
  `--auto-highlights`, `--topic-detection`. Analysis results render automatically
  in human mode (summary, chapters, sentiment, entities, topics, content safety,
  highlights).
- **Customization:** `--word-boost`, `--custom-spelling-file`, `--audio-start`,
  `--audio-end`, `--translate-to`.
- **Webhooks:** `--webhook-url`, `--webhook-auth-header` (`NAME:VALUE`).

Anything without a curated flag is reachable through the escape hatch:
`--config KEY=VALUE` (repeatable) and `--config-file FILE` (a JSON object) accept
any SDK field by its exact name. Precedence is config file < `--config` < explicit
flags.

```sh
aai transcribe call.mp3 \
  --speaker-labels --speakers-expected 2 \
  --redact-pii --redact-pii-policy person_name,phone_number \
  --summarization --summary-type bullets \
  --sentiment-analysis --auto-chapters \
  --config speech_threshold=0.5 \
  --config-file extra.json
```

## Streaming

```sh
aai stream --sample            # stream the hosted wildfires.mp3 sample (same clip as transcribe)
aai stream path/to/audio.wav   # 16 kHz mono WAV streams directly
aai stream path/to/audio.mp3   # other formats need ffmpeg on PATH
aai stream https://…/clip.mp3  # a URL works too (decoded via ffmpeg)
aai stream                     # from the microphone; Ctrl-C to stop
```

`aai stream` exposes the full `StreamingParameters` surface as curated flags:

- **Model & input:** `--speech-model`, `--encoding`, `--language-detection`,
  `--domain`.
- **Turn detection:** `--end-of-turn-confidence-threshold`, `--min-turn-silence`,
  `--max-turn-silence`, `--vad-threshold`, `--format-turns` / `--no-format-turns`,
  `--include-partial-turns`.
- **Features:** `--keyterms-prompt`, `--filter-profanity`, `--speaker-labels`,
  `--max-speakers`, `--voice-focus`, `--voice-focus-threshold`, `--redact-pii`,
  `--redact-pii-policy`, `--redact-pii-sub`, `--inactivity-timeout`,
  `--webhook-url`, `--webhook-auth-header`.

The same escape hatch applies — `--config KEY=VALUE` (repeatable) and
`--config-file FILE` (JSON object) reach any other `StreamingParameters` field,
with precedence config file < `--config` < explicit flags:

```sh
aai stream --sample \
  --max-turn-silence 400 --format-turns \
  --keyterms-prompt "AssemblyAI" \
  --config vad_threshold=0.7
```

## Voice agent

Have a live, two-way voice conversation:

```sh
aai agent                                 # talk; the agent talks back. Ctrl-C to stop.
aai agent --voice james --greeting "Hi"
aai agent --system-prompt-file persona.txt   # load the system prompt from a file
aai agent --list-voices                       # see available voices
```

The agent is full-duplex — your mic stays open while it speaks, so you can interrupt it
mid-sentence (barge-in). **Use headphones**, otherwise the agent hears itself on your
speakers.

## Show the code

Add `--show-code` to `transcribe`, `stream`, or `agent` to print the equivalent Python
SDK code **instead of running** the command — a ready-to-edit starting point for your
own app. It builds the script from exactly the flags you passed, needs no API key
(the generated code reads `ASSEMBLYAI_API_KEY` from the environment), and writes plain
Python to stdout, so you can redirect it straight into a file:

```sh
aai transcribe --sample --speaker-labels --show-code        # print the equivalent script
aai transcribe call.mp3 --sentiment-analysis --show-code > my_transcribe.py
aai stream --show-code                                      # the microphone-streaming idiom
aai agent --voice ivy --show-code                           # the full-duplex agent loop
```

The generated transcribe code includes result handling for the analysis features you
enabled. With `--llm-gateway-prompt` (repeatable — each prompt runs on the previous
response), it emits the chained LLM Gateway calls too:

```sh
aai transcribe call.mp3 \
  --llm-gateway-prompt "summarize" \
  --llm-gateway-prompt "translate the summary to Spanish" \
  --show-code > summarize_then_translate.py
```

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

This project uses [uv](https://docs.astral.sh/uv/). Run tools through `uv run` so they
use the locked environment (`pyproject.toml` + `uv.lock`):

```sh
uv sync --extra dev        # create/refresh the project venv with dev dependencies
uv run aai --help          # run the CLI from the locked environment
uv run pytest              # run the test suite (uv run mypy / ruff likewise)
./scripts/check.sh         # ruff + mypy + pytest (the same checks CI runs on every PR)
```
