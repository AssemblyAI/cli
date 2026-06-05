# Live captions — AssemblyAI Streaming starter

Click record and speak — your microphone streams to AssemblyAI's Streaming v3 API and
transcribes in real time. The browser captures audio and connects **directly** to
AssemblyAI using a one-time token minted by the backend, so your API key never reaches
the client (and no audio is proxied through your server).

## Run locally

```sh
uvicorn api.index:app --reload --port 3000
# open http://localhost:3000  (allow microphone access)
```

`ASSEMBLYAI_API_KEY` is read from `.env` (created for you if you ran `aai init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY` as a
Vercel environment variable (the local `.env` is git-ignored). The backend is just the
`/api/token` function; the WebSocket runs browser → AssemblyAI, so nothing long-running
is needed. `vercel.json` routes the page and the function.

## Ideas to extend

- Add `keyterms_prompt` or a `prompt` for domain vocabulary (the demo uses `u3-rt-pro`).
- Add `format_turns`/punctuation toggles, or speaker labels (`speaker_labels=true`).
- Persist the final transcript, or pipe each finalized turn into the LLM Gateway.
