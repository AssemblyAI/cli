from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE_URL = os.environ.get("ASSEMBLYAI_BASE_URL")
LLM_GATEWAY_URL = os.environ.get(
    "ASSEMBLYAI_LLM_GATEWAY_URL", "https://llm-gateway.assemblyai.com/v1"
)
LLM_MODEL = "claude-haiku-4-5-20251001"

# Public sample so the app works immediately without uploading a local file.
SAMPLE_URL = "https://assembly.ai/wildfires.mp3"

# Main backend customization point. Add, remove, or tune AssemblyAI features here.
TRANSCRIPTION_CONFIG_KWARGS = {
    "speaker_labels": True,
    "auto_chapters": True,
    "sentiment_analysis": True,
    "entity_detection": True,
    "auto_highlights": True,
}
