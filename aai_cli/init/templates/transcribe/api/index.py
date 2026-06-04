"""Transcribe & explore — AssemblyAI starter (FastAPI).

Two short endpoints so each request stays fast (serverless-friendly):
  POST /api/transcribe  -> submit the upload, return {"id": ...}
  GET  /api/status/{id} -> poll; returns the full transcript JSON when complete

The browser (index.html) submits a file, then polls status until done.
Your API key stays on the server — the browser never sees it.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import assemblyai as aai
from assemblyai.api import get_transcript  # single non-blocking GET (see status())
from assemblyai.client import Client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse

load_dotenv()
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY", "")

# Audio-intelligence features to showcase. Tweak these — that's the point of a starter.
CONFIG = aai.TranscriptionConfig(
    speaker_labels=True,
    auto_chapters=True,
    sentiment_analysis=True,
    entity_detection=True,
    auto_highlights=True,
)

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.post("/api/transcribe")
async def transcribe(file: UploadFile) -> dict[str, str]:
    suffix = Path(file.filename or "audio").suffix
    tmp = Path(tempfile.gettempdir()) / f"aai-{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await file.read())
    transcript = aai.Transcriber().submit(str(tmp), config=CONFIG)
    return {"id": transcript.id}


@app.get("/api/status/{transcript_id}")
def status(transcript_id: str) -> dict[str, object]:
    # One non-blocking GET. NOTE: aai.Transcript.get_by_id() BLOCKS until the job
    # finishes (it calls wait_for_completion) — wrong for a poll endpoint, and it
    # would time out on serverless. api.get_transcript does a single request and
    # uses the SDK's configured client (auth + region from aai.settings).
    t = get_transcript(Client.get_default().http_client, transcript_id)
    if t.status == aai.TranscriptStatus.error:
        raise HTTPException(status_code=502, detail=t.error or "Transcription failed")
    if t.status == aai.TranscriptStatus.completed:
        return {"status": "completed", "transcript": t.dict()}
    return {"status": str(getattr(t.status, "value", t.status))}
