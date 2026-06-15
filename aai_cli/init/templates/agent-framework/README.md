# Talk to a cascaded voice agent — AssemblyAI agent-framework starter

Click connect and talk. Unlike the `voice-agent` template (which uses AssemblyAI's
all-in-one Voice Agent API), this app is a **cascade your own backend orchestrates**:
Streaming STT transcribes you, the LLM Gateway generates a reply, and streaming TTS
speaks it back — with turn detection and barge-in handled server-side. The browser
holds one WebSocket to your backend, so your API key never reaches the client.

## Sandbox-only

Streaming TTS has no production host, so the whole cascade runs against the AssemblyAI
sandbox with a sandbox key. Scaffold it that way:

```sh
assembly --sandbox init agent-framework
```

That pins the sandbox hosts in `.env`. Running against production exits with a hint.

## Run locally

```sh
assembly dev   # opens http://localhost:3000 (allow microphone access; headphones recommended)
```

`ASSEMBLYAI_API_KEY` is read from `.env` (created for you by `assembly init`).

## Deploy

This app keeps a **long-running WebSocket**, so it needs a persistent process — not
Vercel's serverless functions. Use the shipped `Procfile`/`Dockerfile` on Render,
Railway, Fly.io, or Google Cloud Run (`gcloud run deploy --source .`):

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

Set `ASSEMBLYAI_API_KEY` and the three sandbox host vars (`ASSEMBLYAI_STREAMING_HOST`,
`ASSEMBLYAI_TTS_HOST`, `ASSEMBLYAI_LLM_GATEWAY_URL`) in the platform's environment.

## Ideas to extend

- Change the `MODEL`, `VOICE`, `SYSTEM_PROMPT`, `GREETING`, or `MAX_HISTORY` in
  `api/settings.py`.
- Replies already stream into TTS sentence-by-sentence as the LLM produces them
  (`_generate_reply` flushes on each `.`/`!`/`?`), and a sliding window of
  `MAX_HISTORY` messages gives the agent memory of the conversation. Tune the
  sentence boundary or `MAX_HISTORY` to trade latency, cost, and recall.
- Add tools (function calling) on the LLM leg so the agent can look things up.
