"""Talk to a cascaded voice agent — AssemblyAI agent-framework starter (FastAPI).

The browser opens one WebSocket to this backend, which runs the cascade itself —
Streaming STT -> LLM Gateway -> streaming TTS — so your API key never reaches the
client. Streaming TTS is sandbox-only, so scaffold with `assembly --sandbox init
agent-framework` and use a sandbox key.

  WS /ws  <- {"type":"input.audio","audio":<b64 pcm>} ; -> transcripts + reply.audio
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import cascade, settings

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Accept the browser socket and run one cascade session over it."""
    await websocket.accept()
    browser = cascade.FastAPIBrowser(websocket)
    await cascade.run_session(browser, cascade.Deps.real(settings))
