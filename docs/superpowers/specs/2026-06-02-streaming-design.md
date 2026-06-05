# `aai stream` — Real-time Streaming Transcription Design

**Date:** 2026-06-02
**Status:** Approved for planning
**Repo:** standalone `assemblyai-cli`
**Builds on:** the v1 CLI (login/transcribe/get/list/samples). Streaming was listed as out-of-scope for v1 in the original design; this is that increment.

## Goal

Add `aai stream` for real-time transcription from either a **microphone** or an **audio file**, using AssemblyAI's v3 streaming API (`assemblyai.streaming.v3`). Success = a user runs `aai stream` and sees their speech transcribed live, or runs `aai stream recording.wav` and watches a file transcribe in real time — and the file path works with no microphone dependency (so it runs in CI and under agents).

## Command

```
aai stream [SOURCE]
   SOURCE                 Optional audio file path. Omit to use the microphone.
   --sample-rate INT      Audio sample rate in Hz (default 16000).
   --device INT           Microphone device index (mic mode only).
   --json                 Emit newline-delimited JSON events instead of live text.
```
Stop with Ctrl-C. Reuses the global `--profile` and existing key resolution.

- **No SOURCE** → microphone capture (requires the optional `[mic]` extra; see Dependencies).
- **SOURCE given** → stream the file through the realtime endpoint, paced to real time. No microphone dependency.

## Dependencies

- **Microphone** path uses PyAudio via the SDK's `aai.extras.MicrophoneStream`. PyAudio is declared as an **optional extra**, not a base dependency:
  ```toml
  [project.optional-dependencies]
  mic = ["pyaudio>=0.2.11"]
  ```
  Installed with `pip install "assemblyai-cli[mic]"`. The base install and CI remain PyAudio-free.
- **File** path:
  - **WAV / PCM** is read with the stdlib `wave` module — zero extra dependency.
  - **Other formats** (mp3, m4a, …) are decoded by piping through an `ffmpeg` subprocess to signed 16-bit little-endian (`s16le`), mono, at the target sample rate. `ffmpeg` is a system tool, not a pip dependency.
- No new base Python dependencies are added.

## Architecture

Two interchangeable **stream sources** behind one interface — an iterator yielding raw PCM `bytes` chunks — both consumed by the SDK's `client.stream(source)`.

```
assemblyai_cli/
  streaming/
    __init__.py
    sources.py        # MicSource, FileSource, and the dependency/format errors
  client.py           # gains stream_audio(...) — the v3 StreamingClient wiring (sole SDK boundary)
  commands/
    stream.py         # the `aai stream` Typer command (thin)
```

### Components

- **`streaming/sources.py`**
  - `MicSource(sample_rate, device)` — context manager / iterator wrapping `aai.extras.MicrophoneStream`. Imports PyAudio lazily; if the import fails, raises `MicDependencyMissing` (mapped to `CLIError`, exit 2, message: `Microphone support isn't installed. Run: pip install 'assemblyai-cli[mic]'`).
  - `FileSource(path, sample_rate)` — iterator yielding fixed-size PCM chunks, sleeping between chunks to match real-time playback duration. WAV/PCM via `wave`; non-WAV via an `ffmpeg` subprocess (`ffmpeg -i <path> -f s16le -acodec pcm_s16le -ac 1 -ar <rate> -`). If the file is not WAV and `ffmpeg` is not on `PATH`, raises `FfmpegMissing` (→ `CLIError`, exit 2, with guidance). A missing file raises `CLIError` (exit 2).
  - Both expose the same iteration contract so the command treats them uniformly.
- **`client.py` → `stream_audio(api_key, source, *, sample_rate, on_begin, on_turn, on_termination, speech_model)`** — the only module importing the SDK. Builds `StreamingClient(StreamingClientOptions(api_key=...))`, registers `StreamingEvents.Begin/Turn/Termination/Error` handlers, calls `connect(StreamingParameters(sample_rate=..., format_turns=True, speech_model=...))` (defaults to `SpeechModel.universal_streaming_multilingual`), then `client.stream(source)`, and always `disconnect(terminate=True)` in a `finally` to flush queued audio and elicit final Turn + Termination events from the server. Streaming `Error` events are collected and raised as `APIError` after disconnect.
- **`commands/stream.py`** — thin: resolves the API key (`config.resolve_api_key`), chooses `FileSource` when a path is given else `MicSource`, defines the render callbacks (below), and calls `client.stream_audio(...)` inside a `try/except KeyboardInterrupt` for a clean Ctrl-C exit.

## Data flow

```
aai stream                      aai stream clip.mp3
  └─ MicSource (PyAudio)          └─ FileSource (wave | ffmpeg → PCM, real-time paced)
        │                               │
        └──────────── PCM byte chunks ──┘
                          │
              client.stream_audio → StreamingClient (v3 ws)
                          │  Begin / Turn / Termination / Error events
                          ▼
                 render callback  ── human: live-updating turn line
                                   └ --json/agent: NDJSON event per line
```

## Output behavior

- **Human / TTY:** the in-progress turn renders as a single line that updates in place; when a turn ends (`end_of_turn`), it is finalized and the next turn starts on a new line. A `Begin` prints a short "listening…" notice; `Termination`/Ctrl-C prints a brief summary.
- **JSON / agent** (`--json`, or piped/CI/agent via the existing `output.resolve_json`): newline-delimited JSON, one object per event, e.g. `{"type": "turn", "transcript": "...", "end_of_turn": true}`, `{"type": "begin", "id": "..."}`, `{"type": "termination"}`. Streaming output bypasses the one-shot `output.emit` (which is for a single result) but uses `resolve_json` for the mode decision.

## Error handling

- Missing `[mic]` extra → `CLIError` exit 2 with the `pip install "assemblyai-cli[mic]"` hint.
- Non-WAV file + no `ffmpeg` → `CLIError` exit 2 with the ffmpeg/WAV guidance.
- Missing/unreadable file → `CLIError` exit 2.
- Unauthenticated → existing `NotAuthenticated` (exit 2).
- Streaming `Error` event → `APIError` (exit 1), surfaced through the command's error path.
- Ctrl-C → graceful `disconnect()` and exit 0.

## Testing

No live microphone or websocket in the automated suite.

- **FileSource (real):** with a tiny generated 16-bit PCM WAV fixture, assert it yields the expected number/size of PCM chunks and that total bytes match the audio length. (Pacing sleeps are patched to no-ops so tests stay fast.)
- **FileSource ffmpeg path:** patch the `ffmpeg` subprocess to a fake emitting known PCM; assert chunks flow. Assert `FfmpegMissing` → `CLIError` when `shutil.which("ffmpeg")` returns None for a non-WAV file.
- **MicSource missing dep:** patch the PyAudio import to raise → assert `MicDependencyMissing` → `CLIError` exit 2 with the install message.
- **Rendering:** a `render_turn(event, *, json_mode)` pure function — fake `Turn`-like events → correct human line vs NDJSON.
- **Command wiring:** patch `client.stream_audio` to drive the registered callbacks with fake `Begin`/`Turn`/`Termination` events → assert human and `--json` output, and that a streaming `Error` yields a non-zero exit.
- **Live mic** test: marked manual / `requires_auth`, not run by default.

## Out of scope (v1 streaming)

Word-level timestamps display, partial-word formatting toggles, PII redaction streaming policy, saving the stream to a file, multi-channel audio, and choosing the speech model — all deferred. The v3 API exposes these; this increment ships mic + file with live turns and JSON events.
