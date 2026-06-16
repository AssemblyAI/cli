"""Per-run configuration for the terminal voice cascade.

Defaults mirror the ``agent-framework`` ``assembly init`` template's
``api/settings.py`` so the CLI conversation and the scaffolded app behave the
same out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass

from aai_cli.agent_framework.voices import DEFAULT_VOICE
from aai_cli.core import llm

DEFAULT_MODEL = llm.DEFAULT_MODEL
DEFAULT_SYSTEM_PROMPT = (
    "You are a friendly, concise voice assistant. Keep replies short and "
    "conversational. Your reply is read aloud by a text-to-speech engine, so "
    "write plain spoken prose — no markdown, emoji, bullet lists, or code."
)
DEFAULT_GREETING = "Hi! I'm your AssemblyAI voice agent. What can I help you with?"
# Sliding-window size: keep the last N messages of conversation as LLM context.
DEFAULT_MAX_HISTORY = 40


@dataclass(frozen=True)
class CascadeConfig:
    """The static knobs for one cascade run, fixed once the flags are parsed."""

    voice: str = DEFAULT_VOICE
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    greeting: str = DEFAULT_GREETING
    model: str = DEFAULT_MODEL
    max_history: int = DEFAULT_MAX_HISTORY
