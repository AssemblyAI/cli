from __future__ import annotations

from assemblyai_cli import llm
from assemblyai_cli.code_gen import serialize, snippets


def render(
    merged: dict[str, object],
    source: str,
    *,
    llm_gateway: dict[str, object] | None = None,
) -> str:
    """Generate a runnable transcribe script reproducing this CLI invocation.

    When `llm_gateway` is given (a dict with ``prompt``/``model``/``max_tokens``), the
    script transforms the transcript through AssemblyAI's LLM Gateway and prints that
    result instead of the analysis sections — mirroring how `--llm-gateway-prompt`
    replaces the normal output.
    """
    if merged:
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=4))
        config_block = f"config = aai.TranscriptionConfig(\n{kwargs}\n)"
        call = f"transcript = transcriber.transcribe({source!r}, config=config)"
    else:
        config_block = ""
        call = f"transcript = transcriber.transcribe({source!r})"

    imports = ["import assemblyai as aai"]
    if llm_gateway:
        imports.append("from openai import OpenAI")

    parts = [
        "import os",
        "",
        *imports,
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
    ]

    if llm_gateway:
        parts += _llm_gateway_block(llm_gateway)
    else:
        parts.append(snippets.result_handling(merged))

    parts.append("")
    return "\n".join(parts)


def _llm_gateway_block(llm_gateway: dict[str, object]) -> list[str]:
    """Emit an OpenAI-compatible LLM Gateway transform over the finished transcript.

    The gateway injects the transcript server-side via the ``transcript_id`` extra-body
    field wherever the ``{{ transcript }}`` tag appears in the prompt.
    """
    content = f"{llm_gateway['prompt']}\n\n{llm.TRANSCRIPT_TAG}"
    return [
        "# Transform the transcript through AssemblyAI's LLM Gateway (OpenAI-compatible).",
        "gateway = OpenAI(",
        '    api_key=os.environ["ASSEMBLYAI_API_KEY"],',
        f"    base_url={llm.GATEWAY_BASE_URL!r},",
        ")",
        "response = gateway.chat.completions.create(",
        f"    model={llm_gateway['model']!r},",
        f'    messages=[{{"role": "user", "content": {content!r}}}],',
        f"    max_tokens={llm_gateway['max_tokens']},",
        '    extra_body={"transcript_id": transcript.id},',
        ")",
        "print(response.choices[0].message.content)",
    ]
