"""Server-side cascade orchestrator for the agent-framework template.

The browser opens one WebSocket to FastAPI and the backend wires three AssemblyAI
primitives together — Streaming STT, the LLM Gateway, and streaming TTS — so every
credential stays on the server. The orchestrator takes injected connect-factories and
an LLM callable (`Deps`) so it runs hermetically against fakes in tests, the same
seam `aai_cli/tts/session.py` uses.

Browser protocol (identical to the voice-agent template):
  in : {"type": "input.audio", "audio": <base64 PCM>}
  out: transcript.user / transcript.agent / reply.audio (base64 in `data`) /
       input.speech.started / reply.done / session.error
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode


def unavailable_reason(settings: Any) -> str | None:
    """Why the cascade can't run, or None when it can.

    Streaming TTS has no production host, so an empty TTS host means the user must
    re-scaffold against the sandbox.
    """
    if not settings.API_KEY:
        return "ASSEMBLYAI_API_KEY is not set — configure it in your deployment's environment."
    if not settings.TTS_HOST:
        return (
            "Streaming TTS has no production host, so this cascade is sandbox-only. "
            "Re-scaffold against the sandbox: assembly --sandbox init agent-framework."
        )
    return None


def stt_url(settings: Any) -> str:
    """The Streaming v3 WebSocket URL with PCM + turn-formatting params."""
    params = urlencode(
        {
            "sample_rate": settings.INPUT_SAMPLE_RATE,
            "encoding": "pcm_s16le",
            "speech_model": "u3-rt-pro",
            "format_turns": "true",
        }
    )
    return f"wss://{settings.STREAMING_HOST}/v3/ws?{params}"


def tts_url(settings: Any) -> str:
    """The streaming-TTS WebSocket URL for the configured voice and sample rate."""
    params = urlencode({"voice": settings.VOICE, "sample_rate": settings.OUTPUT_SAMPLE_RATE})
    return f"wss://{settings.TTS_HOST}/v1/ws/?{params}"


def is_final_user_turn(msg: dict[str, Any]) -> bool:
    """True for a finalized, formatted end-of-turn (the cue to reply)."""
    return bool(msg.get("end_of_turn")) and bool(msg.get("turn_is_formatted"))


def build_messages(system_prompt: str, user_text: str) -> list[dict[str, str]]:
    """The chat `messages` array for one user turn."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
