from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

# Hosts. `assembly init` pins these to the active environment. Streaming TTS only
# exists in the sandbox, so this whole cascade is sandbox-only (see README); the
# defaults point at the sandbox so a bare clone works with a sandbox key.
STREAMING_HOST = os.environ.get(
    "ASSEMBLYAI_STREAMING_HOST", "streaming.sandbox000.assemblyai-labs.com"
)
TTS_HOST = os.environ.get("ASSEMBLYAI_TTS_HOST", "streaming-tts.sandbox000.assemblyai-labs.com")
LLM_GATEWAY_URL = os.environ.get(
    "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.sandbox000.assemblyai-labs.com/v1"
)

# The cascade's knobs — edit these to change behavior.
MODEL = "claude-haiku-4-5-20251001"
VOICE = "ivy"
SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Keep replies short and conversational."
)
GREETING = "Hi! I'm your AssemblyAI voice agent. What can I help you with?"

# 16 kHz PCM in (Streaming v3); 24 kHz PCM out (streaming TTS).
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
