# Agent Notes

This is a buildless FastAPI + browser voice-agent starter. Run it with:

```sh
uvicorn api.index:app --reload --port 3000
```

## Map

- `api/settings.py`: backend token host, token path, WebSocket path, and token expiry.
- `api/index.py`: `/api/token` route. Keep `ASSEMBLYAI_API_KEY` here on the server.
- `static/app.js`: Voice Agent session config, WebSocket lifecycle, UI state, and event handling.
- `static/audio.js`: microphone pipeline, PCM conversion, playback queue, and barge-in helpers.
- `static/styles.css`: visual styling only; the top `:root` block is the primary theme/layout edit point.
- `index.html`: page structure and static asset links. IDs are JavaScript hooks; classes are styling hooks.

## Change Points

- Agent prompt, greeting, voice, audio formats, and microphone constraints: edit `SESSION_CONFIG` in `static/app.js`.
- Backend token lifetime or non-production hosts: edit `api/settings.py`.
- Transcript log rendering: edit `addTurn` in `static/app.js`.
- Playback, barge-in, or PCM conversion: edit `static/audio.js`.
- Visual theme/layout: edit the monotone Vercel-style tokens in `static/styles.css` before changing component rules.
- UI state styling: connection, status, and speaker state use `data-state` or `data-speaker`; prefer CSS changes over JS class rewrites.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `index.html` or `static/`.
- Voice Agent token auth uses `Authorization: Bearer ...` in the backend. This differs from Streaming token auth.
- Voice Agent `greeting` is spoken literally by TTS; write the exact words the user should hear.
- `reply.audio` carries base64 PCM on the `data` field.
- Keep the browser connected directly to AssemblyAI; do not proxy audio through FastAPI unless the user asks.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
