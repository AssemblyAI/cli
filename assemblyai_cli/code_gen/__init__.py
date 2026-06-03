from __future__ import annotations

from assemblyai_cli.code_gen import stream as _stream
from assemblyai_cli.code_gen import transcribe as _transcribe


def transcribe(merged: dict[str, object], source: str) -> str:
    """Generate runnable Python that reproduces this transcribe invocation."""
    return _transcribe.render(merged, source)


def stream(merged: dict[str, object]) -> str:
    """Generate runnable Python that reproduces this streaming invocation."""
    return _stream.render(merged)
