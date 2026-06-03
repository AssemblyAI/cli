from __future__ import annotations

from assemblyai_cli.code_gen import agent as _agent
from assemblyai_cli.code_gen import stream as _stream
from assemblyai_cli.code_gen import transcribe as _transcribe


def agent(voice: str, system_prompt: str, greeting: str) -> str:
    """Generate runnable Python that reproduces this voice-agent session."""
    return _agent.render(voice, system_prompt, greeting)


def transcribe(merged: dict[str, object], source: str) -> str:
    """Generate runnable Python that reproduces this transcribe invocation."""
    return _transcribe.render(merged, source)


def stream(merged: dict[str, object]) -> str:
    """Generate runnable Python that reproduces this streaming invocation."""
    return _stream.render(merged)
