# Live captions — AssemblyAI Streaming starter

Click record and speak — your microphone streams to AssemblyAI's Streaming v3 API and
transcribes in real time. The browser captures audio and connects **directly** to
AssemblyAI using a one-time token minted by the backend, so your API key never reaches
the client (and no audio is proxied through your server). The app uses static
HTML/CSS/JS with no frontend build step.

## Run locally

```sh
assembly dev   # opens http://localhost:3000 (allow microphone access)
```

`ASSEMBLYAI_API_KEY` is read from `.env` (created for you if you ran `assembly init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY` as a
Vercel environment variable (the local `.env` is git-ignored). The shipped `vercel.json`
pins the FastAPI framework preset, so Vercel builds `api/index.py` as the function and
routes every request to that FastAPI app, which serves the page and assets (from
`static/`) plus the `/api/token` route. The WebSocket runs browser → AssemblyAI, so
nothing long-running is needed.

## Deploy elsewhere

The included `Procfile` and `runtime.txt` make this run as a plain Python web app
on Render, Railway, Heroku, Google Cloud Run (`gcloud run deploy --source .`), and
anything else that reads a `Procfile`. Point the platform at this repo and set
`ASSEMBLYAI_API_KEY`; the start command is already declared:

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

On Render, create a **Web Service** connected to your Git repo — it installs
`requirements.txt` and starts via the `Procfile`. (There's no local-directory
deploy; `assembly deploy` covers Vercel/Railway/Fly.)

## Ideas to extend

- Add `keyterms_prompt` or a `prompt` for domain vocabulary in `STREAMING_CONFIG`.
- Add `format_turns`/punctuation toggles, or speaker labels (`speaker_labels=true`) in `STREAMING_CONFIG`.
- Persist the final transcript, or pipe each finalized turn into the LLM Gateway.
