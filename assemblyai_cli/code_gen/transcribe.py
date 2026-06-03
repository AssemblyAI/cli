from __future__ import annotations

from assemblyai_cli.code_gen import serialize, snippets


def render(merged: dict[str, object], source: str) -> str:
    """Generate a runnable transcribe script reproducing this CLI invocation."""
    if merged:
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=4))
        config_block = f"config = aai.TranscriptionConfig(\n{kwargs}\n)"
        call = f"transcript = transcriber.transcribe({source!r}, config=config)"
    else:
        config_block = ""
        call = f"transcript = transcriber.transcribe({source!r})"

    result = snippets.result_handling(merged)

    parts = [
        "import os",
        "",
        "import assemblyai as aai",
        "",
        '# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"',
        'aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]',
        "",
        "transcriber = aai.Transcriber()",
    ]
    if config_block:
        parts += ["", config_block]
    parts += [
        "",
        call,
        "",
        "if transcript.status == aai.TranscriptStatus.error:",
        "    raise RuntimeError(transcript.error)",
        "",
        result,
        "",
    ]
    return "\n".join(parts)
