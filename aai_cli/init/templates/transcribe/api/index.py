"""Transcribe a pre-recorded file — AssemblyAI starter (FastAPI).

Each call stays short, so it's serverless-friendly:
  POST /api/transcribe       -> submit an uploaded file, return {"id": ...}
  POST /api/transcribe-url   -> submit a public URL (defaults to the sample), {"id": ...}
  GET  /api/status/{id}      -> poll; returns the full transcript JSON when complete
  POST /api/ask              -> ask a question about a transcript via the LLM Gateway

The browser (index.html) submits a URL or file, then polls status until done.
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
from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI  # the LLM Gateway is OpenAI-compatible

load_dotenv()
API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
aai.settings.api_key = API_KEY
# Target the same AssemblyAI environment the key was minted for. `aai init` writes
# these for you; leave them unset to use the production defaults.
if base_url := os.environ.get("ASSEMBLYAI_BASE_URL"):
    aai.settings.base_url = base_url
LLM_GATEWAY_URL = os.environ.get(
    "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.assemblyai.com/v1"
)
LLM_MODEL = "claude-haiku-4-5-20251001"

# A public sample so you can try it instantly without uploading anything.
SAMPLE_URL = "https://assembly.ai/wildfires.mp3"

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


def _submit(audio: str) -> dict[str, str]:
    """Submit an audio reference (local file path or public URL); return its id."""
    try:
        transcript = aai.Transcriber().submit(audio, config=CONFIG)
    except Exception as exc:  # missing/invalid key, network, etc. -> clean error, not a 500
        raise HTTPException(
            status_code=502, detail=f"Could not start transcription: {exc}"
        ) from exc
    if transcript.id is None:
        raise HTTPException(status_code=502, detail="Transcription did not return an id")
    return {"id": transcript.id}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile) -> dict[str, str]:
    suffix = Path(file.filename or "audio").suffix
    tmp = Path(tempfile.gettempdir()) / f"aai-{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(await file.read())
    try:
        return _submit(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)  # the upload is sent during submit; don't leave it on disk


@app.post("/api/transcribe-url")
def transcribe_url(url: str = Body(default=SAMPLE_URL, embed=True)) -> dict[str, str]:
    # AssemblyAI transcribes a public URL directly — no upload step needed.
    return _submit(url.strip() or SAMPLE_URL)


@app.post("/api/ask")
def ask(transcript_id: str = Body(...), question: str = Body(...)) -> dict[str, str]:
    """Answer a question about a transcript via the LLM Gateway.

    The gateway is OpenAI-compatible: pass `transcript_id` in `extra_body` and it
    substitutes the `{{ transcript }}` tag with the transcript text server-side.
    """
    client = OpenAI(api_key=API_KEY, base_url=LLM_GATEWAY_URL)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": f"{question}\n\n{{{{ transcript }}}}"}],
            max_tokens=1000,
            extra_body={"transcript_id": transcript_id},
        )
    except Exception as exc:  # surface the gateway's own error (e.g. plan access)
        raise HTTPException(status_code=502, detail=f"LLM Gateway error: {exc}") from exc
    return {"answer": resp.choices[0].message.content or ""}


@app.get("/api/status/{transcript_id}")
def status(transcript_id: str) -> dict[str, object]:
    # One non-blocking GET. NOTE: aai.Transcript.get_by_id() BLOCKS until the job
    # finishes (it calls wait_for_completion) — wrong for a poll endpoint, and it
    # would time out on serverless. api.get_transcript does a single request and
    # uses the SDK's configured client (auth + region from aai.settings).
    try:
        t = get_transcript(Client.get_default().http_client, transcript_id)
    except Exception as exc:  # missing key (ValueError), HTTP/network error -> clean 502
        raise HTTPException(status_code=502, detail=f"Could not fetch transcript: {exc}") from exc
    if t.status == aai.TranscriptStatus.error:
        raise HTTPException(status_code=502, detail=t.error or "Transcription failed")
    if t.status == aai.TranscriptStatus.completed:
        return {"status": "completed", "transcript": t.dict()}
    return {"status": str(getattr(t.status, "value", t.status))}
