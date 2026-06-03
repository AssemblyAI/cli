# AssemblyAI CLI (`aai`) — Demo

The story: **signup → first successful transcript in two commands**, then show it's built for how developers and agents actually work. Paste the blocks in order.

## Setup

Point the CLI at your key. (In a real first-run you'd run `aai login` and paste the key from the browser; this is the repeatable, no-browser path.)

```bash
export ASSEMBLYAI_API_KEY=...   # your key
aai login --api-key "$ASSEMBLYAI_API_KEY"
aai whoami
```

`login` validates the key and stores it in your OS keyring (not a plaintext file); `whoami` confirms which profile is active and that the key reaches the API.

## 1. Zero to transcript in one command

```bash
aai transcribe --sample
```

This transcribes AssemblyAI's hosted sample (`wildfires.mp3`). The submit-and-poll loop is handled for you — one command returns the finished text in a few seconds.

## 2. Built for scripts and agents

```bash
aai transcribe --sample --json | jq
```

The same command emits clean JSON. The CLI auto-detects when it isn't talking to a human (piped, CI, or an AI agent) and switches to JSON automatically — so in your terminal you get readable text, and in a pipeline you get something parseable. (`jq` just pretty-prints; you can drop it.)

## 3. Captions, straight out of the box

```bash
aai transcribe --sample --srt | head -n 8
```

Add `--srt` (or `--vtt`) to get timestamped subtitles from the same command — no extra steps.

## 4. Everything you've run is queryable

```bash
aai list --limit 5
aai get <transcript-id>          # fetch any past transcript by id
```

`list` shows recent transcripts; `get` retrieves one by id (handy for re-fetching a long job later).

## 5. Get into your own code instantly

```bash
aai samples create transcribe
cat transcribe/transcribe.py
python transcribe/transcribe.py
```

`samples create` scaffolds a runnable starter script with your key already wired in — zero edits — and it just runs. (The generated file contains your key, so don't commit it.)

## 6. Real-time streaming (file or microphone)

Stream a file and watch the transcript build in real time — no microphone, no extra dependency (16 kHz mono WAV streams directly; other formats use `ffmpeg`):

```bash
aai stream recording.wav
```

Or transcribe live from your microphone. The mic backend is an optional extra so the base install (and CI) stays lightweight:

```bash
pip install "assemblyai-cli[mic]"
aai stream                       # start talking; Ctrl-C to stop
```

Partial words appear and refine as you speak, finalizing at the end of each turn. Add `--json` for newline-delimited JSON events (one per turn) — ideal for piping into another process or an agent:

```bash
aai stream recording.wav --json
```

Streaming uses AssemblyAI's v3 realtime API under the hood; the CLI just hands it your microphone or file and renders the turns.

---

**The whole onboarding is `aai login` → `aai transcribe`.** Everything else is a convenience on top.

### Tips
- Have `jq` installed for the prettiest JSON output.
- For a *first-time-user* story, use plain `aai login` to show the browser-assisted paste flow instead of `--api-key`.
- `aai logout` clears the stored key when you're done.
