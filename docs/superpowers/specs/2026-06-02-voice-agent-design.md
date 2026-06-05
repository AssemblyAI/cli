# `aai agent` — Voice Agent (speech-in/speech-out) Design

**Date:** 2026-06-02
**Status:** Approved for planning
**Repo:** standalone `assemblyai-cli`
**Builds on:** the v1 CLI (login/transcribe/get/list/samples) and the `aai stream` increment. This adds a live two-way **voice conversation** against AssemblyAI's Voice Agent API.

## Goal

Add `aai agent` for a real-time two-way voice conversation in the terminal: the user speaks into the microphone, the agent replies through the speakers, and the live transcript prints to the screen. Success = a user runs `aai agent`, sees a connect/greeting, talks, and hears the agent talk back, with the conversation transcript on screen.

The Voice Agent API is a single raw WebSocket (`wss://agents.assemblyai.com/v1/ws`) carrying speech in and speech out: PCM16 mono **24 kHz**, base64-encoded. Unlike `aai stream` (which uses the SDK's `StreamingClient`), there is **no AssemblyAI Python SDK** for this endpoint, so we speak the protocol directly over a WebSocket.

## Command

```
aai agent [OPTIONS]
   --voice TEXT          Agent voice (default: ivy). See --list-voices.
   --prompt TEXT         System prompt. Default: a friendly casual-assistant persona.
   --prompt-file PATH    Read the system prompt from a file (overrides --prompt).
   --greeting TEXT       Spoken greeting (default: "Hey, what's on your mind?").
   --full-duplex         Keep the mic open while the agent speaks (true barge-in).
                         Requires headphones. Default is half-duplex.
   --sample-rate INT     Microphone capture rate in Hz (default 24000).
   --device INT          Microphone device index.
   --list-voices         Print known voice IDs and exit.
   --json                Emit newline-delimited JSON events instead of live text.
```

Stop with **Ctrl-C** (clean exit 0, matching `aai stream`). Reuses the global `--profile` and the existing key resolution (`config.resolve_api_key`).

### Echo / duplex behavior

A terminal app has no browser acoustic echo cancellation (AEC), so on open speakers the agent hears its own voice and interrupts itself. The default mitigates this without requiring headphones:

- **Half-duplex (default):** the microphone is *muted* — captured frames are dropped, not sent — for the span between `reply.started` and `reply.done`. The agent therefore never hears its own playback. Barge-in / interruptions are disabled in this mode.
- **`--full-duplex`:** the microphone always streams. Local playback is flushed on `input.speech.started` (and on an interrupted `reply.done`) so barge-in feels snappy. Intended for headphone use; a one-time tip says so.

On start (human mode) the command prints a one-time note explaining the active mode and the headphone recommendation for `--full-duplex`.

## Dependencies

- **Audio I/O** (mic capture *and* speaker playback) uses **PyAudio**, already declared as the optional **`[mic]`** extra (`pip install "assemblyai-cli[mic]"`). PyAudio provides both input and output streams, so no new audio dependency is needed. `aai agent` requires this extra and emits the same friendly "not installed" `CLIError` (exit 2) as `aai stream` when it is missing.
- **WebSocket:** the synchronous client `websockets.sync.client.connect`. `websockets` is already present transitively (the `assemblyai` SDK requires `websockets>=11`); we add an explicit `websockets>=13` to base dependencies to guarantee the sync API (added in 13.0). No new third-party library name is introduced.
- No other new dependencies.

## Architecture

A new `agent/` package mirroring `streaming/`, a thin `commands/agent.py`, and a single WebSocket/protocol boundary in `agent/session.py`.

```
assemblyai_cli/
  agent/
    __init__.py
    session.py     # VoiceAgentSession: WS connect, session.update, receive/dispatch loop
    audio.py       # MicCapture (SDK MicrophoneStream @24k) + Player (PyAudio output) + flush
    render.py      # AgentRenderer: live transcript lines (human) / NDJSON (agent)
    voices.py      # static VOICES list for --list-voices and --voice validation
  commands/agent.py   # the `aai agent` Typer command (thin, like stream.py)
```

### Concurrency model — synchronous WebSocket + threads

Chosen to match the codebase's existing synchronous, blocking-iterator style (`StreamingClient.stream(source)`, `MicrophoneStream`). Three concurrent flows around one `websockets.sync` connection:

1. **Receive loop** (main thread): iterates inbound WS messages and dispatches by `type` to renderer / audio / mute-state callbacks.
2. **Capture thread:** iterates `MicrophoneStream` (blocking), base64-encodes each chunk, and sends `input.audio` — *only after `session.ready`*, and *only when not muted* (half-duplex gating).
3. **Playback thread:** drains a `queue.Queue` of decoded `reply.audio` PCM to the PyAudio output stream.

`send` (capture thread) and `recv` (main thread) act on the connection from different threads; the sync `websockets` connection serializes writes internally, and reads/writes are independent directions, so this is safe. A shared `threading.Event`/flag carries the half-duplex mute state and the "session ready" gate.

**Alternative considered — asyncio + async `websockets`:** one event loop with mic/playback bridged via `run_in_executor`. Rejected: it introduces an async idiom nothing else in the CLI uses and adds real complexity bridging blocking PyAudio in/out to the loop, for no user-visible benefit.

### Components

- **`agent/session.py` → `VoiceAgentSession`** — the only module that opens the WebSocket. Connects with `websockets.sync.client.connect("wss://agents.assemblyai.com/v1/ws", additional_headers={"Authorization": f"Bearer {api_key}"})` using the raw API key — the form shown in the docs' Python `session.resume` example (the browser quickstart's `?token=` query param and the temporary-token endpoint are browser concerns; out of scope). Implementation verifies this header form against a live connect during the first manual test. On open it sends one `session.update` with `system_prompt`, `greeting`, and `output.voice`. It runs the receive loop, dispatching to injected callbacks for each event type; starts/stops the capture and playback threads; and maps WS close codes / `session.error` codes to `APIError`/`CLIError`. Defensive: unknown event types are ignored; `tool.call` is ignored (no tools are configured, so it should not occur).
- **`agent/audio.py`**
  - `MicCapture(sample_rate=24000, device)` — wraps the SDK's `MicrophoneStream`; lazy PyAudio import with the shared `[mic]`-missing `CLIError` (exit 2). Yields PCM byte chunks.
  - `Player(sample_rate=24000)` — opens a PyAudio **output** stream; `enqueue(pcm_bytes)` adds to a queue, a worker thread writes to the stream; `flush()` clears the queue and stops the current write (interruption). `close()` tears down the stream.
- **`agent/render.py` → `AgentRenderer`** — human mode: a speaker indicator plus a two-color transcript flow — user partials (`transcript.user.delta`) update in place and finalize on `transcript.user`; `transcript.agent` prints the agent line. Reuses `StreamRenderer`'s in-place-line (`\r\x1b[K`) technique. JSON mode: NDJSON, one object per event; audio bytes are never emitted.
- **`agent/voices.py`** — a static list of known voice IDs (from the quickstart) backing `--list-voices` and local `--voice` validation (an unknown voice still ultimately surfaces the server's `invalid_value`, but we catch obvious typos early).
- **`commands/agent.py`** — thin: resolves the API key, reads `--prompt-file` if given, builds the session config, constructs `AgentRenderer`/`Player`/`MicCapture`, runs `VoiceAgentSession` inside `try/except KeyboardInterrupt` for a clean Ctrl-C exit. `--list-voices` short-circuits before any connection. Wrapped by the existing `run_command` for CLIError → exit-code mapping.

## Data flow

```
aai agent
  └─ MicCapture (PyAudio @24k) ── PCM chunks ──┐  (gated: after session.ready, unmuted)
                                               ▼
                                   base64 → input.audio  ──►  WS  ◄── session.update (on open)
                                                                │
        server events (session.ready / speech.* / transcript.* / reply.* / error)
                                                                ▼
                                   VoiceAgentSession dispatch
                                     ├─ transcript.* → AgentRenderer (human line / NDJSON)
                                     ├─ reply.audio  → Player.enqueue → PyAudio output
                                     ├─ reply.started → (half-duplex) mute mic
                                     └─ reply.done    → (half-duplex) unmute; if interrupted, Player.flush()
```

## Event handling

Server → client events and their effects:

| Event | Human render | Audio / state |
| --- | --- | --- |
| `session.ready` | "Connected. Speak now. (Ctrl-C to stop)" | open gate; start capture/playback |
| `input.speech.started` | user-speaking indicator | full-duplex: `Player.flush()` |
| `input.speech.stopped` | clear indicator | — |
| `transcript.user.delta` | update user partial line in place | — |
| `transcript.user` | finalize user line | — |
| `reply.started` | agent-speaking indicator | half-duplex: mute mic |
| `reply.audio` | — | `Player.enqueue(decode(data))` |
| `transcript.agent` | print agent line | — |
| `reply.done` | clear indicator | half-duplex: unmute mic; if `status=="interrupted"`: `Player.flush()` |
| `session.error` | error message | map code → APIError/CLIError |

Client → server: `session.update` (once, on open) and `input.audio` (streamed, gated).

## Output behavior

- **Human / TTY:** connect/greeting notice, speaker indicators, in-place user partials finalized per utterance, agent lines printed as they arrive; Ctrl-C prints a brief "Stopped." and exits 0.
- **JSON / agent** (`--json`, or auto via the existing `output.resolve_json`): newline-delimited JSON, one object per server event (e.g. `{"type":"transcript.user","text":"..."}`, `{"type":"transcript.agent","text":"...","interrupted":false}`, `{"type":"reply.done"}`). Audio payloads are omitted. As with streaming, this bypasses the one-shot `output.emit` but uses `resolve_json` for the mode decision.

## Error handling

- Missing `[mic]` extra → `CLIError` exit 2 with the `pip install "assemblyai-cli[mic]"` hint (shared with streaming).
- `UNAUTHORIZED` / `FORBIDDEN` (WS close 1008) or bad key → `CLIError` exit 2, consistent with the CLI's existing auth-error treatment.
- `session.error` (e.g. `invalid_value`, `invalid_config`) or unexpected close (1011) → `APIError` exit 1, surfaced through the command's error path.
- `--prompt-file` unreadable / missing → `CLIError` exit 2.
- Ctrl-C → graceful close of the WebSocket and audio streams, exit 0.

## Testing

No live microphone, speakers, or WebSocket in the automated suite.

- **AgentRenderer:** fake events → assert human lines vs NDJSON, that user partials update in place and finalize, and that audio bytes never appear in JSON output.
- **Session dispatch:** feed a scripted list of fake server messages into the dispatch loop (no real socket) → assert the correct renderer/audio/mute callbacks fire in order, including half-duplex mute on `reply.started`, unmute on `reply.done`, and `Player.flush()` on an interrupted `reply.done` and (full-duplex) on `input.speech.started`.
- **Player:** with a fake PyAudio stream, assert `enqueue` decodes + writes and `flush()` clears the queue / stops current write.
- **MicCapture:** patch the PyAudio/`MicrophoneStream` import to raise → assert the `[mic]`-missing `CLIError` exit 2.
- **Command wiring:** patch `VoiceAgentSession` to drive callbacks with fake events → assert human and `--json` output and exit codes; assert `--list-voices` prints the list and exits without connecting; assert `--prompt-file` is read and overrides `--prompt`.
- **Live conversation** test: marked manual / `requires_auth`, not run by default.

## Out of scope (v1)

Tool / function calling, `reply.create`, `session.resume` / reconnect, temporary-token generation, mid-session reconfiguration, turn-detection tuning flags, output-volume control, Twilio phone integration, file-input mode (a live conversation needs a live mic), and full-duplex gain-ducking. The API supports these; this increment ships a working two-way voice conversation with selectable voice, prompt, and greeting.
