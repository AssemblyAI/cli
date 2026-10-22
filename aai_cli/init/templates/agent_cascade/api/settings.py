from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY: str = os.environ.get("ASSEMBLYAI_API_KEY", "")

# Hosts. `assembly init` pins these to the active environment; the defaults are
# production, so a bare clone works with a production key.
STREAMING_HOST: str = os.environ.get("ASSEMBLYAI_STREAMING_HOST", "streaming.assemblyai.com")
TTS_HOST: str = os.environ.get("ASSEMBLYAI_TTS_HOST", "tts.us.assemblyai.com")
LLM_GATEWAY_URL: str = os.environ.get(
    "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.assemblyai.com/v1"
)

# The cascade's knobs — edit these to change behavior.
MODEL: str = "claude-haiku-4-5-20251001"
# A streaming-TTS voice_id: the preset/library voices are aai-<name>.
VOICE_ID: str = "aai-jane"
SYSTEM_PROMPT: str = (
    "You are a friendly, concise voice assistant. Keep replies short and conversational. "
    "Your reply is read aloud by a text-to-speech engine, so write plain spoken prose — "
    "no markdown, emoji, bullet lists, or code."
)
GREETING: str = "Hi! I'm your AssemblyAI voice agent. What can I help you with?"
MAX_HISTORY: int = 40  # keep the last N messages of conversation context (sliding window)

# 16 kHz PCM in (Streaming v3); 24 kHz PCM out (streaming TTS).
INPUT_SAMPLE_RATE: int = 16000
OUTPUT_SAMPLE_RATE: int = 24000
