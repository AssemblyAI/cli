# Agent Notes

This is a buildless FastAPI + static HTML starter. Run it with:

```sh
uvicorn api.index:app --reload --port 3000
```

## Map

- `api/settings.py`: backend customization for AssemblyAI config, sample URL, and LLM Gateway model.
- `api/index.py`: server routes. Keep `ASSEMBLYAI_API_KEY` here on the server.
- `static/app.js`: browser workflow, polling, tab rendering, and transcript Q&A UI.
- `static/styles.css`: visual styling only.
- `index.html`: page structure and static asset links.

## Change Points

- Transcription features: edit `TRANSCRIPTION_CONFIG_KWARGS` in `api/settings.py`.
- Sample audio URL: edit `SAMPLE_URL` in `api/settings.py` and the matching input value in `index.html`.
- LLM answer behavior: edit `LLM_MODEL` in `api/settings.py` or the `/api/ask` prompt in `api/index.py`.
- Transcript display: edit renderer functions in `static/app.js`.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `index.html` or `static/`.
- Keep every browser `fetch("/api/...")` route registered in `api/index.py`.
- Keep `/api/status/{transcript_id}` non-blocking; do not use SDK helpers that wait for completion in that polling route.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
