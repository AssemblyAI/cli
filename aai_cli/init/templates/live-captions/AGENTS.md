# Agent Notes

This is a buildless FastAPI + browser microphone starter. Run it with:

```sh
aai dev
```

## Map

- `api/settings.py`: backend token host, token path, WebSocket path, and token expiry.
- `api/index.py`: `/api/token` route. Keep `ASSEMBLYAI_API_KEY` here on the server.
- `public/static/app.js`: browser state, WebSocket lifecycle, and Streaming API params.
- `public/static/audio.js`: microphone pipeline and PCM downsampling helpers.
- `public/static/styles.css`: visual styling only; the top `:root` block is the primary theme/layout edit point.
- `public/index.html`: page structure and static asset links. IDs are JavaScript hooks; classes are styling hooks.

## Change Points

- Streaming model, sample rate, encoding, and turn formatting: edit `STREAMING_CONFIG` in `public/static/app.js`.
- Backend token lifetime or non-production hosts: edit `api/settings.py`.
- Caption rendering: edit `onMessage` in `public/static/app.js`.
- Microphone/downsampling behavior: edit `public/static/audio.js`.
- Visual theme/layout: edit the monotone Vercel-style tokens in `public/static/styles.css` before changing component rules.
- UI state styling: record and status state use `data-state`; prefer CSS changes over JS class rewrites.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `public/index.html` or `public/static/`.
- Streaming token auth uses the raw API key in the backend `Authorization` header, not `Bearer`.
- Keep the browser connected directly to AssemblyAI; do not proxy the audio stream through FastAPI unless the user asks.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
