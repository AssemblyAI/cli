# Agent Notes

This is a buildless FastAPI + browser microphone starter. Run it with:

```sh
uvicorn api.index:app --reload --port 3000
```

## Map

- `api/settings.py`: backend token host, token path, WebSocket path, and token expiry.
- `api/index.py`: `/api/token` route. Keep `ASSEMBLYAI_API_KEY` here on the server.
- `static/app.js`: browser state, WebSocket lifecycle, and Streaming API params.
- `static/audio.js`: microphone pipeline and PCM downsampling helpers.
- `static/styles.css`: visual styling only.
- `index.html`: page structure and static asset links.

## Change Points

- Streaming model, sample rate, encoding, and turn formatting: edit `STREAMING_CONFIG` in `static/app.js`.
- Backend token lifetime or non-production hosts: edit `api/settings.py`.
- Caption rendering: edit `onMessage` in `static/app.js`.
- Microphone/downsampling behavior: edit `static/audio.js`.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `index.html` or `static/`.
- Streaming token auth uses the raw API key in the backend `Authorization` header, not `Bearer`.
- Keep the browser connected directly to AssemblyAI; do not proxy the audio stream through FastAPI unless the user asks.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
