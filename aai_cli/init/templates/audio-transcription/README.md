# Transcribe a pre-recorded file — AssemblyAI starter

Transcribe an audio/video URL (defaults to a public sample) or upload a file, and
see the transcript with speaker labels, chapters, sentiment, entities, and
highlights. Built with FastAPI + static HTML/CSS/JS. There is no frontend build step.

## Run locally

```sh
uvicorn api.index:app --reload --port 3000
# open http://localhost:3000
```

`ASSEMBLYAI_API_KEY` is read from `.env` (already created for you if you ran `aai init`).

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY`
as a Vercel environment variable (the local `.env` is git-ignored and not deployed).
No extra config is needed: Vercel discovers the FastAPI app in `api/index.py`,
which serves the page and its `static/` assets itself.

## Ideas to extend

- Show chapter summaries and highlight timestamps.
- Add a waveform / audio player synced to the transcript.
- Swap the analysis features in `TRANSCRIPTION_CONFIG_KWARGS` (`api/settings.py`).
- Change transcript rendering in `static/app.js`.
