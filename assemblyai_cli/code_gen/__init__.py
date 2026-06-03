from __future__ import annotations

from assemblyai_cli.code_gen import transcribe as _transcribe


def transcribe(merged: dict[str, object], source: str) -> str:
    return _transcribe.render(merged, source)
