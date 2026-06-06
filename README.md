# AssemblyAI CLI

Transcribe. Stream. Converse. — speech AI from your terminal.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh
```

The installer uses [`pipx`](https://pipx.pypa.io) when available (falling back to
`pip --user`) and requires Python 3.11+. Prefer to do it yourself:

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

## Build an app

`aai init` is how you **build a new app** — it scaffolds a complete project from a
template. This is the starting point whenever you want to *create* something, including
a voice agent app.

```sh
aai init                            # pick a template, scaffold it, install deps, open the browser
aai init audio-transcription myapp  # non-interactive: template + directory
aai init voice-agent my-agent       # build a voice agent app (full FastAPI + browser starter)
```

`aai init` copies a small, self-contained FastAPI + HTML project you can run locally
and deploy to Vercel as-is. Your key is written to a git-ignored `.env` (and is never
sent to the browser). Use `--no-install` to scaffold only.

> **Building a voice agent? Use `aai init voice-agent`, not `aai agent`.** `aai agent`
> only *runs* a live mic conversation in the terminal and writes no code; `aai init`
> creates the actual app.

## API key & security

`aai` resolves your key in this order:

1. The `ASSEMBLYAI_API_KEY` environment variable.
2. The OS keyring (macOS Keychain, Windows Credential Manager, Linux Secret
   Service), written only when you run `aai login`.

Two things worth knowing: the key is **never stored in a plaintext dotfile** —
`aai login` puts it in the OS keyring, and the only on-disk config (`config.toml`)
holds just profile names. And there is **no `--api-key` flag on run commands**
(`transcribe`, `stream`, …), so a key can't leak into `ps` output or shell history
via a command's arguments.

**Prefer not to persist the key at all?** Skip `aai login` and set the environment
variable instead — it's checked *before* the keyring, so nothing is ever written to
disk:

```sh
ASSEMBLYAI_API_KEY=sk_... aai transcribe call.mp3
```

Prefixing it on a single command (rather than `export`-ing it) scopes the secret to
that one process. To also keep it out of your shell history, inject it from a secret
manager at call time:

```sh
# 1Password CLI
ASSEMBLYAI_API_KEY=$(op read "op://Private/AssemblyAI/api key") aai transcribe call.mp3
op run -- aai transcribe call.mp3                          # …or wrap the whole command

# HashiCorp Vault
ASSEMBLYAI_API_KEY=$(vault kv get -field=key secret/assemblyai) aai stream

# macOS Keychain (a generic-password item you manage)
ASSEMBLYAI_API_KEY=$(security find-generic-password -w -s assemblyai -a "$USER") aai transcribe call.mp3
```

In CI, set `ASSEMBLYAI_API_KEY` as a masked secret — nothing is stored. The env var
also overrides a stored key for one-off use; `aai logout` purges the keyring entry,
and `aai whoami` / `aai doctor` confirm which source is active without printing the key.

## Commands

| Command | What it does |
| --- | --- |
| `aai login` / `logout` / `whoami` | Manage the stored API key. |
| `aai doctor` | Check your environment is ready (API key, network, ffmpeg, microphone, agent tooling). |
| `aai transcribe <file\|url>` | Transcribe an audio file, URL, or YouTube URL (`--sample` for a demo, `--llm` to transform the result through LLM Gateway, `--show-code` to print the equivalent Python). |
| `aai transcripts list` / `get <id>` | Browse and fetch past transcripts. |
| `aai stream [file]` | Real-time transcription from a file or the microphone. |
| `aai agent` | *Run* a live two-way voice conversation (to **build** a voice agent app, use `aai init voice-agent`). |
| `aai llm <prompt>` | Prompt AssemblyAI's LLM Gateway (over a past transcript with `--transcript-id`, or a live streamed transcript with `--follow`). |
| `aai setup install` | Set up your coding agent for AssemblyAI (docs MCP + skills). |
| `aai samples create <name>` | Scaffold a runnable starter script (reads your key from `ASSEMBLYAI_API_KEY`). |
| `aai keys list` / `create` / `rename` | Manage your API keys (browser login). |
| `aai balance` / `usage` / `limits` | Account billing, usage, and rate limits (browser login). |
| `aai sessions list` / `get <id>` | Browse past streaming (real-time) sessions (browser login). |
| `aai audit` | View your account's audit log (browser login). |

Add `--json` to any command for machine-readable output (it's also the default when
output is piped or run by an agent). Errors always go to **stderr**, so stdout stays
clean for pipelines. Auth problems surface as a clean "not authenticated" error
across every command.

> **Tip:** Quote URLs that contain `?` (most YouTube links do). In zsh the `?` is a
> glob character, so an unquoted URL fails with `zsh: no matches found` before the
> command runs:
>
> ```sh
> aai transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
> ```

## Account self-service

These commands use your browser login session (run `aai login` without
`--api-key`), not your API key:

```sh
aai keys list                       # list API keys (masked) across projects
aai keys create --name ci-pipeline  # mint a new key (printed once)
aai keys rename 123 "prod"          # relabel a key

aai balance                         # remaining account balance
aai usage --start 2026-05-01 --end 2026-06-01
aai limits                          # rate limits per service

aai sessions list --status completed
aai sessions get <session-id>       # one streaming session's details

aai audit --limit 20                # recent account audit-log entries
aai audit --action token.create     # filter by action
```

If a command reports it needs a browser login, your session has expired — run
`aai login` again. (AMS sessions are short-lived and cannot be refreshed
silently.)

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
aai stream --system-audio      # macOS: system/app audio + mic as separate sessions
aai stream --system-audio-only # macOS: system/app audio without the mic
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

On macOS, `--system-audio` uses ScreenCaptureKit to capture system/app audio
without a loopback driver and streams it in a separate Streaming session from
the microphone. The default terminal UI labels finalized turns as `You:` or
`System:`. The first run may ask for Screen & System Audio Recording and
Microphone permissions. The helper does not record screen frames, but macOS
still uses that combined permission label for native system audio capture.
`--system-audio-only` skips the microphone.

## Live transcript → live LLM

`aai stream --llm "PROMPT"` runs a prompt over the live transcript through LLM Gateway,
refreshing the answer on every finalized turn — one command, no pipe to wire up:

```sh
aai stream --llm "summarize action items as I talk"
```

It's repeatable, so prompts chain — each runs on the previous one's response:

```sh
aai stream --llm "extract action items" --llm "rewrite them as a checklist"
```

On a terminal you watch one evolving panel; piped onward it emits one JSON object per
refresh (`{"turns": N, "output": "…"}`). Ctrl-C to stop.

**Prefer the pipe?** The same thing composes from the primitives: `aai stream -o text`
writes one finalized turn per line, and `aai llm -f` (`--follow`) re-runs your prompt
over the *growing* transcript. Reach for this when you want a `--system` prompt or other
tools in the pipeline:

```sh
aai stream -o text | aai llm -f --system "You are a meeting scribe" "summarize action items as I talk"
```

Without `--follow`, `aai llm` stays one-shot — it reads stdin to EOF and answers once
(`cat notes | aai llm "summarize"`).

## Voice agent

Have a live, two-way voice conversation. (To **build** a voice agent *app*, use
`aai init voice-agent` instead — this command just runs a conversation in the terminal.)

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
enabled. With `--llm` (repeatable — each prompt runs on the previous response), it emits
the chained LLM Gateway calls too:

```sh
aai transcribe call.mp3 \
  --llm "summarize" \
  --llm "translate the summary to Spanish" \
  --show-code > summarize_then_translate.py
```

`aai stream --llm "…" --show-code` likewise emits the live transcribe→LLM-per-turn loop.

## Pipelines

`aai` is built to compose with the rest of your shell. Output is machine-clean
(errors go to stderr), commands read `-` from stdin, and `-o`/`--output` prints a
single field so you rarely need `jq`.

**Pick one field with `-o`:**

```sh
aai transcribe call.mp3 -o text        # just the transcript text
aai transcribe call.mp3 -o id          # just the transcript id
aai transcribe call.mp3 -o utterances  # speaker-labeled lines
aai transcribe video.mp4 -o srt        # SubRip (.srt) captions
aai transcribe call.mp3 -o json | jq .  # full JSON when you do want jq
```

**Read audio from stdin (`-`):**

```sh
ffmpeg -i talk.mp4 -f wav - | aai transcribe -        # transcribe any video
curl -sL https://example.com/ep.mp3 | aai transcribe -  # no temp file
ffmpeg -i in.mp4 -f s16le -ac 1 -ar 16000 - | aai stream -   # live, from a pipe
```

**Feed text into the LLM Gateway** (`aai llm` reads piped stdin). For a transcript,
`aai transcribe --llm "…"` does it in one step — the pipe is for any *other* text:

```sh
cat notes.txt | aai llm "turn these into a changelog"
```

**Pipe a live stream into other tools.** For live LLM summaries use `aai stream --llm`
(above) — one process, clean Ctrl-C. To pipe the live transcript into a *different* tool,
note that a Ctrl-C in a pipe hits both sides, so to stop the producer and let the
consumer finish, signal only the producer — or end the stream on its own:

```sh
# end after 30s by signaling just the producer (macOS: brew install coreutils, use gtimeout)
timeout -s INT 30s aai stream -o text | grep -i "action item"

# or end on a natural pause (server-side inactivity timeout, in seconds)
aai stream -o text --inactivity-timeout 5 > call.txt

# capture then process (most robust)
aai stream -o text > call.txt        # Ctrl-C to stop
aai llm "summarize" < call.txt
```

## Recipes

A cookbook of `aai` composed with common Unix tools. macOS shown; on Linux swap
`pbcopy`/`pbpaste` → `xclip -sel clip`/`xclip -o` and `say` → `spd-say`.

**Chain `aai llm` into other tools** with `-o text` — it prints just the answer, so it
pipes onward cleanly (no `jq` needed):

```sh
aai transcribe call.mp3 -o text | aai llm -o text "list action items" | pbcopy
```

**`aai llm` is a general text filter** — it reads stdin, audio optional:

```sh
git log --oneline -30 | aai llm "write release notes grouped by feature/fix"
cat error.log         | aai llm "what's the root cause and the one-line fix?"
```

**Translate a sample, then port the generated code** — `--show-code` prints the Python
for the pipeline you described, and `aai llm` rewrites it in another language:

```sh
aai transcribe --sample --llm "translate to french" --show-code | aai llm "rewrite in rust"
```

**Mine the analysis JSON with `jq`** — enable a feature, then slice `-o json`:

```sh
aai transcribe call.mp3 --sentiment-analysis -o json | jq -r '.sentiment_analysis_results[] | "\(.sentiment)\t\(.text)"'
aai transcribe call.mp3 --entity-detection  -o json | jq -r '.entities[] | "\(.entity_type): \(.text)"' | sort -u
```

**Pick a past transcript with `fzf`, then summarize it:**

```sh
aai transcripts list --json \
  | jq -r '.[] | "\(.id)\t\(.status)\t\(.created)"' \
  | fzf | cut -f1 \
  | xargs -I{} aai llm "summarize the key decisions" --transcript-id {}
```

**Who talked the most** (speaker-labeled utterances + `awk`):

```sh
aai transcribe call.mp3 --speaker-labels -o utterances | awk -F: '{print $1}' | sort | uniq -c | sort -rn
```

**Redact PII before it leaves your machine:**

```sh
aai transcribe call.mp3 --redact-pii --redact-pii-policy person_name,phone_number,email_address -o text | pbcopy
```

**Caption a YouTube video (sing-along subtitles)** — download the video, transcribe it
to SubRip with `-o srt`, then burn the captions in with ffmpeg. These steps pass *files*
to each other (not stdin/stdout), and ffmpeg's `subtitles` filter needs a seekable file,
so chain them with `&&` rather than `|` — each step runs only if the previous succeeds:

```sh
URL="https://www.youtube.com/watch?v=6YzGOq42zLk&list=RD6YzGOq42zLk&start_radio=1"

yt-dlp --no-playlist -f 'bv*+ba/b' --merge-output-format mp4 -o video.mp4 "$URL" && aai transcribe video.mp4 -o srt > captions.srt && ffmpeg -i video.mp4 -vf "subtitles=captions.srt" -c:a copy out.mp4
```

`--no-playlist` matters for music links: the `&list=RD…` suffix is an autoplay radio, so
without it yt-dlp downloads an endless mix instead of the one video. This burns in
**static per-line captions** — for true word-by-word karaoke highlighting you'd render an
ASS subtitle file from the transcript's word timings (`-o json` → `words[]`) instead.

**DIY voice assistant** — speak a question, hear the answer (use headphones):

```sh
aai stream -o text | while IFS= read -r line; do
  echo "$line" | aai llm -o text "answer in one short sentence" | say
done
```

## AI coding agents

Set your coding agent up for AssemblyAI — the live docs (MCP server), the AssemblyAI
skill, and the bundled `aai-cli` skill — so your agent writes current, correct
integration code:

```sh
aai setup install        # docs MCP + assemblyai skill + bundled aai-cli skill (user scope)
aai setup status         # show what's set up
aai setup remove         # unwind all three
```

`install` shells out to `claude mcp add` for the MCP and `npx skills add` for the
`assemblyai` skill; the `aai-cli` skill ships inside the package and is copied in
directly (no network). Pass `--scope project` to scope the MCP server to the current
project. A missing `claude` or `npx` is reported and skipped (with the manual command to
run), not treated as an error.

## Development

This project uses [uv](https://docs.astral.sh/uv/). Run tools through `uv run` so they
use the locked environment (`pyproject.toml` + `uv.lock`):

```sh
uv sync --extra dev        # create/refresh the project venv with dev dependencies
uv run aai --help          # run the CLI from the locked environment
uv run pytest              # run the test suite (uv run mypy / ruff likewise)
./scripts/check.sh         # full lint/typecheck/test/build gate CI runs on every PR
```

## License

This project is licensed under the [MIT License](LICENSE).
