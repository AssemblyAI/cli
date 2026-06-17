from __future__ import annotations

from aai_cli.code_gen import serialize, snippets
from aai_cli.code_gen.serialize import GatewayOptions
from aai_cli.core import environments, llm, youtube

# ``-o/--output`` choice -> printed-result code, mirroring the run path's
# ``client._FIELD_RENDERERS`` semantics: plain fields, the speaker-labeled
# utterances loop, the SRT/VTT export endpoints, and the raw ``json_response`` payload.
_OUTPUT_SNIPPETS: dict[str, str] = {
    "text": "print(transcript.text)",
    "id": "print(transcript.id)",
    "status": "print(transcript.status.value)",
    "utterances": (
        'for utt in transcript.utterances or []:\n    print(f"Speaker {utt.speaker}: {utt.text}")'
    ),
    "srt": "print(transcript.export_subtitles_srt())",
    "vtt": "print(transcript.export_subtitles_vtt())",
    "json": "print(json.dumps(transcript.json_response, default=str))",
}

# The subtitle exports take the --chars-per-caption knob as a kwarg.
_SUBTITLE_FORMATS = ("srt", "vtt")


def render(
    merged: dict[str, object],
    source: str,
    *,
    llm_gateway: GatewayOptions | None = None,
    output: str | None = None,
    chars_per_caption: int | None = None,
    download_sections: list[str] | None = None,
) -> str:
    """Generate a runnable transcribe script reproducing this CLI invocation.

    When `llm_gateway` is given (a dict with ``prompt``/``model``/``max_tokens``), the
    script transforms the transcript through AssemblyAI's LLM Gateway and prints that
    result instead of the analysis sections — mirroring how `--llm-gateway-prompt`
    replaces the normal output.

    When `output` (a ``-o/--output`` field name) is given, the script prints that one
    field instead — and, as in the real command, it takes precedence over the LLM chain
    and the analysis sections. `chars_per_caption` shapes the srt/vtt export calls.

    When `download_sections` (yt-dlp ``--download-sections`` specs) is given for a
    downloadable URL, the generated yt-dlp call fetches only those parts of the source.
    """
    if output is not None:
        llm_gateway = None  # `-o` returns before the chain runs in the real command
    needs_download = youtube.is_downloadable_url(source)
    # Sections only apply to the download path; ignore them for a local file.
    sections = list(download_sections) if (needs_download and download_sections) else []
    ranges_expr, needs_re = _download_ranges(sections)
    parts = (
        _header_block(
            llm_gateway,
            output,
            needs_download=needs_download,
            needs_re=needs_re,
            has_sections=ranges_expr is not None,
        )
        + _transcribe_block(merged, source, needs_download=needs_download, ranges_expr=ranges_expr)
        + _result_block(merged, llm_gateway, output, chars_per_caption)
    )
    parts.append("")
    return "\n".join(parts)


def _render_seconds(value: float) -> str:
    """A Python literal for a section bound (``inf`` has no bare literal form)."""
    if value == float("inf"):
        return "float('inf')"
    if value == float("-inf"):
        return "float('-inf')"
    return repr(value)


def _download_ranges(sections: list[str]) -> tuple[str | None, bool]:
    """Render a ``download_range_func(...)`` expression for `sections`.

    Returns ``(expression_or_None, needs_re_import)`` — ``None`` when there are no
    sections, and the flag is true when a chapter-regex spec means the generated
    script needs ``import re``.
    """
    if not sections:
        return None, False
    chapters, ranges, from_url = youtube.parse_download_sections(sections)
    chapters_src = "[" + ", ".join(f"re.compile({c!r})" for c in chapters) + "]"
    ranges_src = (
        "[" + ", ".join(f"({_render_seconds(s)}, {_render_seconds(e)})" for s, e in ranges) + "]"
    )
    return f"download_range_func({chapters_src}, {ranges_src}, {from_url})", bool(chapters)


def _header_block(
    llm_gateway: GatewayOptions | None,
    output: str | None,
    *,
    needs_download: bool,
    needs_re: bool,
    has_sections: bool,
) -> list[str]:
    """Imports plus the api-key (and non-default environment) settings lines."""
    stdlib_imports = ["import os"]
    if needs_download:
        # The download path fetches audio to a temp dir before uploading.
        stdlib_imports.append("import tempfile")
    if needs_re:
        # Chapter-regex sections compile to re.compile(...) in the generated script.
        stdlib_imports.append("import re")
    if output == "json":
        stdlib_imports.append("import json")
    stdlib_imports.sort()
    imports = ["import assemblyai as aai"]
    if needs_download:
        imports.append("import yt_dlp")
    if has_sections:
        # --download-sections builds yt-dlp's download_ranges via this helper.
        imports.append("from yt_dlp.utils import download_range_func")
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


def _ydl_options_lines(ranges_expr: str | None) -> list[str]:
    """The yt-dlp options dict for the download — one line, or multi-line when sections
    add a ``download_ranges``/``force_keyframes_at_cuts`` pair."""
    if ranges_expr is None:
        return ['        {"format": "bestaudio/best", "outtmpl": f"{_tmp}/%(id)s.%(ext)s"}']
    return [
        "        {",
        '            "format": "bestaudio/best",',
        '            "outtmpl": f"{_tmp}/%(id)s.%(ext)s",',
        f'            "download_ranges": {ranges_expr},',
        '            "force_keyframes_at_cuts": True,',
        "        }",
    ]


def _transcribe_block(
    merged: dict[str, object], source: str, *, needs_download: bool, ranges_expr: str | None
) -> list[str]:
    """The transcriber setup, optional config, the transcribe call, and error check."""
    parts = ["", "transcriber = aai.Transcriber()"]
    config_arg = ""
    if merged:
        kwargs = "\n".join(serialize.config_kwarg_lines(merged, indent=4))
        parts += ["", f"config = aai.TranscriptionConfig(\n{kwargs}\n)"]
        config_arg = ", config=config"
    if needs_download:
        # AssemblyAI can't read a YouTube/podcast page URL itself, so download the
        # audio with yt-dlp into a temp dir and upload the local file — what the CLI does.
        parts += [
            "",
            "# AssemblyAI can't fetch this page URL itself; download the audio first.",
            "with tempfile.TemporaryDirectory() as _tmp:",
            "    with yt_dlp.YoutubeDL(",
            *_ydl_options_lines(ranges_expr),
            "    ) as _ydl:",
            f"        _info = _ydl.extract_info({source!r}, download=True)",
            "        _audio = _ydl.prepare_filename(_info)",
            f"    transcript = transcriber.transcribe(_audio{config_arg})",
        ]
    else:
        parts += ["", f"transcript = transcriber.transcribe({source!r}{config_arg})"]
    return [
        *parts,
        "",
        "if transcript.status == aai.TranscriptStatus.error:",
        "    raise RuntimeError(transcript.error)",
        "",
    ]


def _result_block(
    merged: dict[str, object],
    llm_gateway: GatewayOptions | None,
    output: str | None,
    chars_per_caption: int | None,
) -> list[str]:
    """The printed-result lines: one ``-o`` field, the LLM chain, or the analysis sections."""
    if output is not None:
        if output in _SUBTITLE_FORMATS and chars_per_caption is not None:
            return [
                f"print(transcript.export_subtitles_{output}"
                f"(chars_per_caption={chars_per_caption}))"
            ]
        # Unknown names fall back to the plain text, like select_transcript_field does.
        return [_OUTPUT_SNIPPETS.get(output, _OUTPUT_SNIPPETS["text"])]
    if llm_gateway:
        return _llm_gateway_block(llm_gateway)
    return [snippets.result_handling(merged)]


def _llm_gateway_block(llm_gateway: GatewayOptions) -> list[str]:
    """Emit a chained OpenAI-compatible LLM Gateway transform over the transcript.

    The generated script loops over the prompts: the first runs over the transcript
    (injected server-side via ``transcript_id`` wherever the ``{{ transcript }}`` tag
    appears), and each subsequent prompt runs over the previous response.
    """
    prompts = llm_gateway["prompts"]
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
