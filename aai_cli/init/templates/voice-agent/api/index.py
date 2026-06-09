"""Talk to a voice agent — AssemblyAI Voice Agent starter (FastAPI).

The browser connects directly to AssemblyAI's Voice Agent WebSocket (speech-in → LLM →
speech-out, with built-in turn detection and TTS) using a short-lived, single-use token
minted here — so your API key never reaches the client.

  POST /api/token -> {"token": ..., "ws_url": "wss://.../v1/ws"}
"""

from __future__ import annotations

from pathlib import Path

# httpx2 is Pydantic's maintained fork of httpx (API-identical, just renamed) — not a
# typo. Keep the "2"; see requirements.txt.
import httpx2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import settings

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
app = FastAPI()
app.mount("/static", StaticFiles(directory=PUBLIC / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")


@app.post("/api/token")
def token() -> dict[str, str]:
    """Mint a one-time Voice Agent token. The browser uses it to open the WebSocket."""
    # NOTE: the Voice Agent token uses Bearer auth (unlike the streaming token).
    try:
        resp = httpx2.get(
            f"https://{settings.AGENTS_HOST}{settings.TOKEN_PATH}",
            params={"expires_in_seconds": settings.TOKEN_EXPIRES_IN_SECONDS},
            headers={"Authorization": f"Bearer {settings.API_KEY}"},
        )
        resp.raise_for_status()
    except Exception as exc:  # missing/invalid key, network -> clean 502, not a 500
        raise HTTPException(
            status_code=502, detail=f"Could not mint voice-agent token: {exc}"
        ) from exc
    return {
        "token": resp.json()["token"],
        "ws_url": f"wss://{settings.AGENTS_HOST}{settings.WEBSOCKET_PATH}",
    }
