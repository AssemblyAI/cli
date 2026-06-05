"""Talk to a voice agent — AssemblyAI Voice Agent starter (FastAPI).

The browser connects directly to AssemblyAI's Voice Agent WebSocket (speech-in → LLM →
speech-out, with built-in turn detection and TTS) using a short-lived, single-use token
minted here — so your API key never reaches the client.

  POST /api/token -> {"token": ..., "ws_url": "wss://.../v1/ws"}
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

load_dotenv()
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
# `aai init` writes this for you; defaults to production. (No scheme — host only.)
AGENTS_HOST = os.environ.get("ASSEMBLYAI_AGENTS_HOST", "agents.assemblyai.com")

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.post("/api/token")
def token() -> dict[str, str]:
    """Mint a one-time Voice Agent token. The browser uses it to open the WebSocket."""
    # NOTE: the Voice Agent token uses Bearer auth (unlike the streaming token).
    try:
        resp = httpx2.get(
            f"https://{AGENTS_HOST}/v1/token",
            params={"expires_in_seconds": 60},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        resp.raise_for_status()
    except Exception as exc:  # missing/invalid key, network -> clean 502, not a 500
        raise HTTPException(
            status_code=502, detail=f"Could not mint voice-agent token: {exc}"
        ) from exc
    return {"token": resp.json()["token"], "ws_url": f"wss://{AGENTS_HOST}/v1/ws"}
