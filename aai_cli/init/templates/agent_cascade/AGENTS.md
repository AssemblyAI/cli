# Agent Notes

This is a buildless FastAPI + browser starter for a **cascaded** voice agent
(Streaming STT -> LLM Gateway -> streaming TTS), orchestrated server-side. Run it with:

```sh
assembly dev
```

## Map

- `api/settings.py`: API key, hosts, model, voice, system prompt, greeting, sample rates.
- `api/cascade.py`: the orchestrator — STT/TTS socket helpers, the LLM stream, turn
  detection, barge-in, and the `/ws` browser adapter. Built with injected `Deps` so it
  is tested against fakes.
- `api/index.py`: FastAPI app — serves the page/assets and the `/ws` WebSocket.
- `static/app.js`: WebSocket lifecycle, mic capture, UI state, and event handling
  (`_CONFIG` block at the top is the primary edit point).
- `static/audio.js`: microphone pipeline, PCM conversion, playback queue, barge-in.
- `static/styles.css`: visual styling only; the top `:root` block is the theme edit point.
- `static/index.html`: page structure and static asset links.

## Change Points

- Model, voice, prompt, greeting, sample rates: edit `api/settings.py`.
- Cascade behavior (turn detection, barge-in, LLM->TTS piping): edit `api/cascade.py`.
- Transcript log rendering: edit `addTurn` in `static/app.js`.
- Playback, barge-in, or PCM conversion: edit `static/audio.js`.

## Invariants

- Never expose `ASSEMBLYAI_API_KEY` or any server secret in `static/`.
- Streaming TTS is sandbox-only; keep this app pointed at the sandbox hosts.
- `reply.audio` carries base64 PCM on the `data` field.
- The browser <-> backend event protocol matches the `voice-agent` template — keep it
  stable so `static/audio.js` and the UI stay reusable.
- Keep the app buildless unless the user explicitly asks for a frontend toolchain.
