"""Transcribe a pre-recorded file — AssemblyAI starter (FastAPI).

Each call stays short, so it's serverless-friendly:
  POST /api/transcribe       -> submit an uploaded file, return {"id": ...}
  POST /api/transcribe-url   -> submit a public URL (defaults to the sample), {"id": ...}
  GET  /api/status/{id}      -> poll; returns the full transcript JSON when complete
  POST /api/ask              -> ask a question about a transcript via the LLM Gateway

The browser (static/index.html + static/app.js) submits a URL or file, then
polls status.
Your API key stays on the server — the browser never sees it.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import assemblyai as aai
from assemblyai.api import get_transcript  # single non-blocking GET (see status())
from assemblyai.client import Client
from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI  # the LLM Gateway is OpenAI-compatible

from . import settings

aai.settings.api_key = settings.API_KEY
# Target the same AssemblyAI environment the key was minted for. `assembly init` writes
# this for you; leave it unset to use the production default.
if settings.ASSEMBLYAI_BASE_URL:
    aai.settings.base_url = settings.ASSEMBLYAI_BASE_URL

CONFIG = aai.TranscriptionConfig(**settings.TRANSCRIPTION_CONFIG_KWARGS)

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


def _submit(audio: str) -> dict[str, str]:
    """Submit an audio reference (local file path or public URL); return its id."""
    try:
        transcript = aai.Transcriber().submit(audio, config=CONFIG)
    except Exception as exc:  # missing/invalid key, network, etc. -> clean error
        raise HTTPException(
            status_code=502, detail=f"Could not start transcription: {exc}"
        ) from exc
    if transcript.id is None:
        raise HTTPException(status_code=502, detail="Transcription did not return an id")
    return {"id": transcript.id}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile) -> dict[str, str]:
    _require_key()
    suffix = Path(file.filename or "audio").suffix
    tmp = Path(tempfile.gettempdir()) / f"aai-{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        return _submit(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)  # the upload is sent during submit; don't leave it on disk


@app.post("/api/transcribe-url")
def transcribe_url(url: str = Body(default=settings.SAMPLE_URL, embed=True)) -> dict[str, str]:
    _require_key()
    # AssemblyAI transcribes a public URL directly — no upload step needed.
    return _submit(url.strip() or settings.SAMPLE_URL)


@app.post("/api/ask")
def ask(transcript_id: str = Body(...), question: str = Body(...)) -> dict[str, str]:
    """Answer a question about a transcript via the LLM Gateway."""
    _require_key()
    client = OpenAI(api_key=settings.API_KEY, base_url=settings.LLM_GATEWAY_URL)
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": f"{question}\n\n{{{{ transcript }}}}"}],
            max_tokens=1000,
            extra_body={"transcript_id": transcript_id},
        )
    except Exception as exc:  # surface the gateway's own error (e.g. plan access)
        raise HTTPException(status_code=502, detail=f"LLM Gateway error: {exc}") from exc
    return {"answer": resp.choices[0].message.content or ""}


@app.get("/api/status/{transcript_id}")
def status(transcript_id: str) -> dict[str, object]:
    _require_key()
    # One non-blocking GET. NOTE: aai.Transcript.get_by_id() BLOCKS until the job
    # finishes (it calls wait_for_completion), which is wrong for a poll endpoint.
    try:
        t = get_transcript(Client.get_default().http_client, transcript_id)
    except Exception as exc:  # missing key (ValueError), HTTP/network error -> clean 502
        raise HTTPException(status_code=502, detail=f"Could not fetch transcript: {exc}") from exc
    if t.status == aai.TranscriptStatus.error:
        raise HTTPException(status_code=502, detail=t.error or "Transcription failed")
    if t.status == aai.TranscriptStatus.completed:
        return {"status": "completed", "transcript": t.dict()}
    return {"status": str(getattr(t.status, "value", t.status))}
