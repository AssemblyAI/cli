# Agent Notes

This is a buildless FastAPI + static HTML starter. Run it with:

```sh
uvicorn api.index:app --reload --port 3000
```

## Map

- `api/settings.py`: backend customization for AssemblyAI config, sample URL, and LLM Gateway model.
- `api/index.py`: server routes. Keep `ASSEMBLYAI_API_KEY` here on the server.
- `public/static/app.js`: browser workflow, polling, tab rendering, and transcript Q&A UI.
- `public/static/styles.css`: visual styling only; the top `:root` block is the primary theme/layout edit point.
- `public/index.html`: page structure and static asset links. IDs are JavaScript hooks; classes are styling hooks.

## Change Points

- Transcription features: edit `TRANSCRIPTION_CONFIG_KWARGS` in `api/settings.py`.
- Sample audio URL: edit `SAMPLE_URL` in `api/settings.py` and the matching input value in `public/index.html`.
- LLM answer behavior: edit `LLM_MODEL` in `api/settings.py` or the `/api/ask` prompt in `api/index.py`.
- Transcript display: edit renderer functions in `public/static/app.js`.
- Visual theme/layout: edit the monotone Vercel-style tokens in `public/static/styles.css` before changing component rules.
- UI state styling: status, tabs, and sentiment use `data-state`, `.is-active`, or `data-sentiment`; prefer CSS changes over JS class rewrites.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `public/index.html` or `public/static/`.
- Keep every browser `fetch("/api/...")` route registered in `api/index.py`.
- Keep `/api/status/{transcript_id}` non-blocking; do not use SDK helpers that wait for completion in that polling route.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
