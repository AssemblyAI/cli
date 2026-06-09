# Live captions — AssemblyAI Streaming starter

Click record and speak — your microphone streams to AssemblyAI's Streaming v3 API and
transcribes in real time. The browser captures audio and connects **directly** to
AssemblyAI using a one-time token minted by the backend, so your API key never reaches
the client (and no audio is proxied through your server). The app uses static
HTML/CSS/JS with no frontend build step.

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
is needed.

## Deploy elsewhere

The included `Procfile` and `runtime.txt` make this run as a plain Python web app
on Render, Railway, Heroku, Google Cloud Run (`gcloud run deploy --source .`), and
anything else that reads a `Procfile`. Point the platform at this repo and set
`ASSEMBLYAI_API_KEY`; the start command is already declared:

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

## Ideas to extend

- Add `keyterms_prompt` or a `prompt` for domain vocabulary in `STREAMING_CONFIG`.
- Add `format_turns`/punctuation toggles, or speaker labels (`speaker_labels=true`) in `STREAMING_CONFIG`.
- Persist the final transcript, or pipe each finalized turn into the LLM Gateway.
