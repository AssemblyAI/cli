# Transcribe & explore — AssemblyAI starter

Upload an audio/video file and see the transcript with speaker labels, chapters,
sentiment, entities, and highlights. Built with FastAPI + a single HTML page.

## Run locally

```sh
uvicorn api.index:app --reload --port 3000
# open http://localhost:3000
```

`ASSEMBLYAI_API_KEY` is read from `.env` (already created for you if you ran `aai init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY`
as a Vercel environment variable (the local `.env` is git-ignored and not deployed).
No extra config — `vercel.json` routes the page and the `/api` function.

## Ideas to extend

- Show chapter summaries and highlight timestamps.
- Add a waveform / audio player synced to the transcript.
- Swap the analysis features in `CONFIG` (api/index.py).
