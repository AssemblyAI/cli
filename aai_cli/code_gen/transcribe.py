from __future__ import annotations

from typing import cast

from aai_cli import environments, llm
from aai_cli.code_gen import serialize, snippets

# ``-o/--output`` choice -> printed-result code, mirroring the run path's
# ``client._FIELD_RENDERERS`` semantics: plain fields, the speaker-labeled
# utterances loop, the SRT export endpoint, and the raw ``json_response`` payload.
_OUTPUT_SNIPPETS: dict[str, str] = {
    "text": "print(transcript.text)",
    "id": "print(transcript.id)",
    "status": "print(transcript.status.value)",
    "utterances": (
        'for utt in transcript.utterances or []:\n    print(f"Speaker {utt.speaker}: {utt.text}")'
    ),
    "srt": "print(transcript.export_subtitles_srt())",
    "json": "print(json.dumps(transcript.json_response, default=str))",
}


def render(
    merged: dict[str, object],
    source: str,
    *,
    llm_gateway: dict[str, object] | None = None,
    output: str | None = None,
) -> str:
    """Generate a runnable transcribe script reproducing this CLI invocation.

    When `llm_gateway` is given (a dict with ``prompt``/``model``/``max_tokens``), the
    script transforms the transcript through AssemblyAI's LLM Gateway and prints that
    result instead of the analysis sections — mirroring how `--llm-gateway-prompt`
    replaces the normal output.

    When `output` (a ``-o/--output`` field name) is given, the script prints that one
    field instead — and, as in the real command, it takes precedence over the LLM chain
    and the analysis sections.
    """
    if output is not None:
        llm_gateway = None  # `-o` returns before the chain runs in the real command
    parts = (
        _header_block(llm_gateway, output)
        + _transcribe_block(merged, source)
        + _result_block(merged, llm_gateway, output)
    )
    parts.append("")
    return "\n".join(parts)


def _header_block(llm_gateway: dict[str, object] | None, output: str | None) -> list[str]:
    """Imports plus the api-key (and non-default environment) settings lines."""
    stdlib_imports = ["import os"]
    if output == "json":
        stdlib_imports.insert(0, "import json")
    imports = ["import assemblyai as aai"]
    if llm_gateway:
        imports.append("from openai import OpenAI")
    parts = [
        *stdlib_imports,
        "",
        *imports,
        "",
        '# Export your key first:  export ASSEMBLYAI_API_KEY="<your key>"',
        'aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]',
    ]
    # The SDK ships pointing at production, so only a non-default environment
    # (e.g. --sandbox) needs its api base spelled out in the generated script.
    env = environments.active()
    if env.api_base != environments.get(environments.DEFAULT_ENV).api_base:
        parts.append(f"aai.settings.base_url = {env.api_base!r}")
    return parts


def _transcribe_block(merged: dict[str, object], source: str) -> list[str]:
    """The transcriber setup, optional config, the transcribe call, and error check."""
    parts = ["", "transcriber = aai.Transcriber()"]
    if merged:
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=4))
        parts += ["", f"config = aai.TranscriptionConfig(\n{kwargs}\n)"]
        call = f"transcript = transcriber.transcribe({source!r}, config=config)"
    else:
        call = f"transcript = transcriber.transcribe({source!r})"
    return parts + [
        "",
        call,
        "",
        "if transcript.status == aai.TranscriptStatus.error:",
        "    raise RuntimeError(transcript.error)",
        "",
    ]


def _result_block(
    merged: dict[str, object], llm_gateway: dict[str, object] | None, output: str | None
) -> list[str]:
    """The printed-result lines: one ``-o`` field, the LLM chain, or the analysis sections."""
    if output is not None:
        # Unknown names fall back to the plain text, like select_transcript_field does.
        return [_OUTPUT_SNIPPETS.get(output, _OUTPUT_SNIPPETS["text"])]
    if llm_gateway:
        return _llm_gateway_block(llm_gateway)
    return [snippets.result_handling(merged)]


def _llm_gateway_block(llm_gateway: dict[str, object]) -> list[str]:
    """Emit a chained OpenAI-compatible LLM Gateway transform over the transcript.

    The generated script loops over the prompts: the first runs over the transcript
    (injected server-side via ``transcript_id`` wherever the ``{{ transcript }}`` tag
    appears), and each subsequent prompt runs over the previous response.
    """
    prompts = cast("list[str]", llm_gateway["prompts"])
    prompt_lines = "\n".join(f"    {p!r}," for p in prompts)
    return [
        "# Transform the transcript through AssemblyAI's LLM Gateway (OpenAI-compatible).",
        "# Each prompt runs on the previous response; the first runs on the transcript.",
        "gateway = OpenAI(",
        '    api_key=os.environ["ASSEMBLYAI_API_KEY"],',
        f"    base_url={environments.active().llm_gateway_base!r},",
        ")",
        "prompts = [",
        prompt_lines,
        "]",
        "result = None",
        "for i, prompt in enumerate(prompts):",
        "    if i == 0:",
        f'        content = prompt + "\\n\\n{llm.TRANSCRIPT_TAG}"',
        '        extra = {"transcript_id": transcript.id}',
        "    else:",
        '        content = prompt + "\\n\\nTranscript:\\n" + result',
        "        extra = None",
        "    response = gateway.chat.completions.create(",
        f"        model={llm_gateway['model']!r},",
        '        messages=[{"role": "user", "content": content}],',
        f"        max_tokens={llm_gateway['max_tokens']},",
        "        extra_body=extra,",
        "    )",
        "    result = response.choices[0].message.content",
        '    print(f"Step {i + 1}: {prompt}")',
        "    print(result)",
    ]
