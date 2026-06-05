"""Live captions — AssemblyAI Streaming starter (FastAPI).

The browser streams microphone audio directly to AssemblyAI's Streaming v3 WebSocket
using a short-lived, single-use token minted here — so your API key never reaches the
client.

  POST /api/token -> {"token": ..., "ws_url": "wss://.../v3/ws"}
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

load_dotenv()
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# `aai init` writes this for you; defaults to production. (No scheme — host only.)
STREAMING_HOST = os.environ.get("ASSEMBLYAI_STREAMING_HOST", "streaming.assemblyai.com")

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.post("/api/token")
def token() -> dict[str, str]:
    """Mint a one-time streaming token. The browser uses it to open the WebSocket."""
    # NOTE: the streaming token uses the raw API key as Authorization (no 'Bearer').
    try:
        resp = httpx.get(
            f"https://{STREAMING_HOST}/v3/token",
            params={"expires_in_seconds": 60},
            headers={"Authorization": API_KEY},
        )
        resp.raise_for_status()
    except Exception as exc:  # missing/invalid key, network -> clean 502, not a 500
        raise HTTPException(
            status_code=502, detail=f"Could not mint streaming token: {exc}"
        ) from exc
    return {"token": resp.json()["token"], "ws_url": f"wss://{STREAMING_HOST}/v3/ws"}
