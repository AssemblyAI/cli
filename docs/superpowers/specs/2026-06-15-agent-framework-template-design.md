# `agent-framework` init template — design

**Date:** 2026-06-15
**Status:** Approved (design); pending implementation plan

## Goal

Add a fourth `assembly init` starter template, `agent-framework`, that delivers the
same browser UI/UX as the existing `voice-agent` template but is built on a
**cascaded** architecture instead of AssemblyAI's all-in-one Voice Agent endpoint.
The cascade wires three primitives together server-side:

1. **Streaming STT** (v3 realtime WebSocket) — speech in, turn detection.
2. **LLM Gateway** (OpenAI-compatible HTTP) — reply generation.
3. **Streaming TTS** (sandbox WebSocket) — speech out.

This is the "framework" you would build yourself if the managed Voice Agent did not
exist, so it is a useful, instructive starter for users who want control over each leg.

## Architecture

```
Browser ──mic PCM (16k)──►  FastAPI /ws  ──audio bytes──►  STT WS (v3)
        ◄──transcripts────         │       ◄──Turn/end_of_turn──┘
        ◄──reply.audio (24k)──     ├──finalized turn──►  LLM Gateway (OpenAI-compatible, streamed)
                                   └──reply text──►  TTS WS (sandbox) ──Audio──► back to browser
```

The browser opens **one** same-origin WebSocket to our FastAPI backend. The backend
runs the full cascade and keeps all three API credentials server-side. No token mint
is needed (unlike `voice-agent`/`live-captions`, which mint short-lived tokens because
the browser connects directly to AssemblyAI).

### Browser ↔ backend protocol (identical to `voice-agent`)

Reusing the existing event vocabulary keeps `static/audio.js` unchanged and reduces
`static/app.js` to a connection-setup change.

- Browser → server:
  - `{type: "input.audio", audio: <base64 PCM>}` — one mic frame.
  - `{type: "session.update", session: {...}}` — optional; the backend may apply
    `system_prompt`/`greeting`/`voice` overrides or ignore it. Kept for parity.
- Server → browser:
  - `{type: "transcript.user", text}` — STT transcript (partial and final).
  - `{type: "transcript.agent", text}` — the LLM reply text.
  - `{type: "reply.audio", data: <base64 PCM>}` — a TTS audio chunk.
  - `{type: "input.speech.started"}` — barge-in: user started talking; browser stops
    queued audio.
  - `{type: "reply.done", status}` — reply finished (or `interrupted`).
  - `{type: "session.error", message}` — any leg failed; surfaced in the UI.

## Components (template files)

- `api/index.py` — FastAPI app. Serves `index.html` + `/static`, and exposes
  `@app.websocket("/ws")` which hands each accepted connection to the orchestrator.
- `api/settings.py` — config from env: `ASSEMBLYAI_API_KEY`, `ASSEMBLYAI_STREAMING_HOST`,
  `ASSEMBLYAI_TTS_HOST`, `ASSEMBLYAI_LLM_GATEWAY_URL`, model (`claude-haiku-4-5-20251001`),
  voice (`ivy`), system prompt, greeting, sample rates (16 kHz in, 24 kHz out). Fails
  fast with an actionable message when `ASSEMBLYAI_TTS_HOST` is empty (production has no
  streaming-TTS host).
- `api/cascade.py` — per-session async orchestrator:
  - Opens the STT WS (API key auth) and forwards mic bytes from the browser.
  - Reads STT `Turn` events: emits `transcript.user` for partials; on `end_of_turn`
    (formatted final) triggers the reply pipeline.
  - Reply pipeline: streams the LLM completion (emitting `transcript.agent`), pipes the
    reply text into a TTS WS (Begin → Generate → ForceFlushTextBuffer → Audio frames →
    Terminate, mirroring `aai_cli/tts/session.py`), and forwards each Audio frame as
    `reply.audio`.
  - Barge-in: a new non-empty user partial while a reply is in flight emits
    `input.speech.started` and cancels the in-flight LLM/TTS task.
  - Speaks the configured greeting on connect (greeting text → TTS → `reply.audio`).
  - Tears down cleanly on browser disconnect / socket close / LLM error, cancelling
    sibling tasks.
- `static/index.html` — copy of `voice-agent`'s page with the eyebrow/title/subtitle
  reworded to describe the cascade; IDs/classes unchanged.
- `static/styles.css` — identical to `voice-agent`.
- `static/audio.js` — identical to `voice-agent` (mic pipeline, PCM player, downsample,
  base64 helpers).
- `static/app.js` — same event handling as `voice-agent`; `connect()` opens a same-origin
  `/ws` directly (no `/api/token` fetch).
- Scaffold parity files: `README.md`, `AGENTS.md`, `env.example`, `gitignore`,
  `requirements.txt` (adds `websockets` + `openai` to the FastAPI/uvicorn base),
  `Procfile`, `Dockerfile`, `dockerignore`, `runtime.txt`, `vercel.json`.

## Stack

Async throughout: the `websockets` async client (STT + TTS), `openai.AsyncOpenAI`
pointed at the gateway base (streamed completion), and FastAPI/Starlette WebSockets for
the browser side. Served as a long-lived process by `uvicorn`.

## CLI wiring (shared edits — unavoidable for a new template)

- `aai_cli/init/templates.py` — add `"agent-framework": "Agent Framework"` to `TEMPLATES`
  and to `TEMPLATE_ORDER`.
- `aai_cli/app/init_exec.py` — add `"ASSEMBLYAI_TTS_HOST": env.streaming_tts_host` to
  `_active_env_vars()`. This appends one extra (unused, empty-in-prod) var to every
  template's `.env`; harmless to the others and required by `agent-framework`.

These are the standard registration touch-points for a template; the "a new command
edits no shared file" rule applies to commands, not templates.

## Deploy / operational caveats

- **Sandbox-only.** Streaming TTS has no production host (`streaming_tts_host` is empty
  in `production`). A credential is valid only against the environment that minted it,
  so the *entire* cascade must point at `sandbox000` with a sandbox key. The README
  leads with `assembly --sandbox init agent-framework`, which pins all three hosts to
  sandbox via `_active_env_vars()`. Running against production exits fast with a
  `--sandbox` hint.
- **Not Vercel-serverless.** The persistent browser WebSocket needs a long-lived
  process, so the primary deploy path is the shipped `Procfile`/`Dockerfile` (Render,
  Railway, Fly, Cloud Run). `vercel.json` is retained for static parity, but the README
  is explicit that the WebSocket requires a long-running host.

## Error handling

Every leg maps a failure to a single `session.error` event to the browser (mirroring
`voice-agent`). The orchestrator cancels sibling tasks on browser disconnect, STT/TTS
socket close, or LLM error, so a session never leaks tasks or sockets.

## Testing

The parametrized init-template contract tests (`tests/test_init_template_*.py`) cover
the new template automatically once it is in `TEMPLATE_ORDER`: required files present,
renamed dotfiles (`gitignore` → `.gitignore`, `env.example`), wheel packaging, and
ruff/prettier cleanliness. The plan will confirm exactly what those contracts assert and
add template-specific coverage where needed (notably the new `ASSEMBLYAI_TTS_HOST` env
var and the prod fail-fast path).

## Out of scope (YAGNI)

- Function calling / tools on the LLM leg (left as an "ideas to extend" note).
- Sentence-level TTS streaming tuning beyond what is needed for acceptable latency.
- A production TTS path (does not exist yet).
