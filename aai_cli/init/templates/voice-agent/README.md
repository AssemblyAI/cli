# Talk to a voice agent — AssemblyAI Voice Agent starter

Click connect and talk to a voice agent: speech in → LLM → speech out, with built-in
turn detection, TTS, and barge-in. The browser connects **directly** to AssemblyAI's
Voice Agent WebSocket using a one-time token minted by the backend, so your API key
never reaches the client and no audio is proxied through your server. The app uses
static HTML/CSS/JS with no frontend build step.

## Run locally

```sh
aai dev   # opens http://localhost:3000 (allow microphone access; headphones recommended)
```

`ASSEMBLYAI_API_KEY` is read from `.env` (created for you if you ran `aai init`).
The Voice Agent API requires a plan with access enabled.

## Deploy to Vercel

Push this folder to a Git repo and import it on Vercel. Set `ASSEMBLYAI_API_KEY` as a
Vercel environment variable (the local `.env` is git-ignored). The backend is just the
`/api/token` function; the WebSocket runs browser → AssemblyAI, so nothing long-running
is needed.

## Deploy elsewhere

The included `Procfile` and `runtime.txt` make this run as a plain Python web app
on Render, Railway, Heroku, Google Cloud Run (`gcloud run deploy --source .`), and
anything else that reads a `Procfile`. Point the platform at this repo and set
`ASSEMBLYAI_API_KEY`; the start command is already declared:

```sh
uvicorn api.index:app --host 0.0.0.0 --port $PORT
```

## Ideas to extend

- Change the `greeting`, `systemPrompt`, or `voice` in `SESSION_CONFIG` (`public/static/app.js`).
- Add tools (function calling) so the agent can look things up or take actions.
- Tune `input.turn_detection` (`min_silence`/`max_silence`) inside `SESSION_CONFIG`.
