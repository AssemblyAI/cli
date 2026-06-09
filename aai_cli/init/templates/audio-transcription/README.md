# Transcribe a pre-recorded file — AssemblyAI starter

Transcribe an audio/video URL (defaults to a public sample) or upload a file, and
see the transcript with speaker labels, chapters, sentiment, entities, and
highlights. Built with FastAPI + static HTML/CSS/JS. There is no frontend build step.

## Run locally

```sh
aai dev   # installs deps if needed, starts the server, opens http://localhost:3000
```

`ASSEMBLYAI_API_KEY` is read from `.env` (already created for you if you ran `aai init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY`
as a Vercel environment variable (the local `.env` is git-ignored and not deployed).
No extra config is needed (no `vercel.json`): Vercel runs `api/index.py` as the
function, and that FastAPI app serves both the page and assets (from `static/`)
and the API.

## Deploy elsewhere

The included `Procfile` and `runtime.txt` make this run as a plain Python web app
on Render, Railway, Heroku, Google Cloud Run (`gcloud run deploy --source .`), and
anything else that reads a `Procfile`. Point the platform at this repo and set
`ASSEMBLYAI_API_KEY`; the start command is already declared:

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

## Ideas to extend

- Show chapter summaries and highlight timestamps.
- Add a waveform / audio player synced to the transcript.
- Swap the analysis features in `TRANSCRIPTION_CONFIG_KWARGS` (`api/settings.py`).
- Change transcript rendering in `static/app.js`.
