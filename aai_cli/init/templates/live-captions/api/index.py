"""Live captions — AssemblyAI Streaming starter (FastAPI).

The browser streams microphone audio directly to AssemblyAI's Streaming v3 WebSocket
using a short-lived, single-use token minted here — so your API key never reaches the
client.

  POST /api/token -> {"token": ..., "ws_url": "wss://.../v3/ws"}
"""

from __future__ import annotations

from pathlib import Path

from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import settings

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _require_key() -> None:
    """Fail fast with an actionable message when the API key isn't configured."""
    if not settings.API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ASSEMBLYAI_API_KEY is not set — configure it in your deployment's environment variables.",
        )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/token")
def token() -> dict[str, str]:
    """Mint a one-time streaming token via the AssemblyAI SDK. The browser uses it to open the WebSocket."""
    _require_key()
    try:
        client = StreamingClient(
            StreamingClientOptions(api_key=settings.API_KEY, api_host=settings.STREAMING_HOST)
        )
        token = client.create_temporary_token(expires_in_seconds=settings.TOKEN_EXPIRES_IN_SECONDS)
    except Exception as exc:  # missing/invalid key, network -> clean 502, not a 500
        raise HTTPException(
            status_code=502, detail=f"Could not mint streaming token: {exc}"
        ) from exc
    return {
        "token": token,
        "ws_url": f"wss://{settings.STREAMING_HOST}{settings.WEBSOCKET_PATH}",
    }
