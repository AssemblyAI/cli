"""Example-based code_gen tests: serializers, snippets, transcribe rendering.

Stream/agent scaffold tests live in test_code_gen_stream_agent.py; hypothesis
fuzz/property tests live in test_code_gen_fuzz.py.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from aai_cli.code_gen import serialize
from aai_cli.code_gen.transcribe import render as render_transcribe_code


def test_py_literal_basic_types():
    assert serialize.py_literal("en_us") == "'en_us'"
    assert serialize.py_literal(True) == "True"
    assert serialize.py_literal(42) == "42"
    assert serialize.py_literal(["a", "b"]) == "['a', 'b']"
    assert (
        serialize.py_literal({"AssemblyAI": ["assembly ai"]}) == "{'AssemblyAI': ['assembly ai']}"
    )


def test_py_literal_speech_model_enum():
    from assemblyai.streaming.v3 import SpeechModel

    assert serialize.py_literal(SpeechModel.u3_rt_pro) == "SpeechModel.u3_rt_pro"


def test_config_kwarg_lines_emits_indented_kwargs():
    lines = serialize.config_kwarg_lines(
        {"speaker_labels": True, "language_code": "en_us"}, indent=4
    )
    assert lines == ["    speaker_labels=True,", "    language_code='en_us',"]


def test_config_kwarg_lines_empty_dict():
    assert serialize.config_kwarg_lines({}, indent=4) == []


from aai_cli.code_gen import snippets  # noqa: E402


def test_result_handling_includes_only_enabled_features():
    out = snippets.result_handling({"speaker_labels": True, "sentiment_analysis": True})
    assert "transcript.utterances" in out  # speaker_labels
    assert "transcript.sentiment_analysis" in out
    assert "transcript.summary" not in out  # summarization not enabled


def test_result_handling_default_prints_text():
    out = snippets.result_handling({})
    assert out.strip() == "print(transcript.text)"


def test_every_render_feature_has_a_snippet():
    # Maintainability tripwire. CONTRACT: each analysis feature rendered by a
    # `_render_<name>` function in transcribe_render.py must have a snippet whose
    # name == <name>. `_render_text` is excluded (it renders the flat transcript and,
    # inline, the speaker_labels utterances). `speaker_labels` therefore has a snippet
    # but no `_render_speaker_labels` function, so it is an allowed orphan.
    import inspect

    from aai_cli import transcribe_render

    rendered = {
        name[len("_render_") :]
        for name, _ in inspect.getmembers(transcribe_render, inspect.isfunction)
        if name.startswith("_render_") and name != "_render_text"
    }
    covered = set(snippets.SNIPPET_FEATURES)
    ORPHANS = {"speaker_labels"}  # rendered inside _render_text, not its own function

    missing = rendered - covered
    assert not missing, f"render features without a snippet: {missing}"
    unexpected_orphans = covered - rendered - ORPHANS
    assert not unexpected_orphans, f"snippets with no matching renderer: {unexpected_orphans}"


import ast  # noqa: E402

from aai_cli import code_gen  # noqa: E402


def test_transcribe_render_parses_and_uses_env_key():
    code = code_gen.transcribe({"speaker_labels": True}, source="https://assembly.ai/wildfires.mp3")
    ast.parse(code)  # raises SyntaxError if malformed
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code
    assert "https://assembly.ai/wildfires.mp3" in code
    assert "transcript.utterances" in code  # result handling for speaker_labels
    assert "{{API_KEY}}" not in code  # never echo a real key
    # config kwargs are rendered 4-space indented inside the TranscriptionConfig call
    assert "aai.TranscriptionConfig(\n    speaker_labels=True,\n)" in code


def test_transcribe_render_no_config_is_minimal():
    code = code_gen.transcribe({}, source="audio.mp3")
    ast.parse(code)
    assert "print(transcript.text)" in code
    assert "TranscriptionConfig(" not in code  # no kwargs -> no config object


def test_transcribe_render_youtube_downloads_before_upload():
    # AssemblyAI can't fetch a YouTube watch URL itself, so the generated script must
    # download the audio with yt-dlp first and upload the local file (mirroring the CLI),
    # not hand the raw URL to transcribe() — which would fail with a download error.
    code = code_gen.transcribe({}, source="https://www.youtube.com/watch?v=ZRcpnM26nJM")
    ast.parse(code)
    assert "import yt_dlp" in code
    assert "import tempfile" in code
    assert "yt_dlp.YoutubeDL(" in code
    assert "extract_info('https://www.youtube.com/watch?v=ZRcpnM26nJM', download=True)" in code
    # The transcribe call takes the downloaded local path, never the YouTube URL.
    assert "transcriber.transcribe(_audio)" in code
    assert "transcribe('https://www.youtube.com" not in code
    assert 'transcribe("https://www.youtube.com' not in code


def test_transcribe_render_podcast_page_downloads_before_upload():
    # Podcast episode pages (extractor-matched, like YouTube) generate the same
    # download-first script; the page URL never reaches transcribe().
    url = "https://podcasts.apple.com/us/podcast/some-show/id1535809341?i=1000123456789"
    code = code_gen.transcribe({}, source=url)
    ast.parse(code)
    assert "import yt_dlp" in code
    assert f"extract_info({url!r}, download=True)" in code
    assert "transcriber.transcribe(_audio)" in code
    assert "transcribe('https://podcasts.apple.com" not in code


def test_transcribe_render_youtube_passes_config_to_local_upload():
    # With a config object the download still wraps the upload, and config flows through.
    code = code_gen.transcribe({"speaker_labels": True}, source="https://youtu.be/abc123")
    ast.parse(code)
    assert "transcriber.transcribe(_audio, config=config)" in code


def test_transcribe_render_download_sections_timestamp_range():
    # --download-sections renders yt-dlp's download_ranges; infinities have no bare
    # literal form, so they render as float('inf')/float('-inf'). A timestamp-only spec
    # needs no `import re`, and force_keyframes_at_cuts pins the cut to exact times.
    code = code_gen.transcribe(
        {},
        source="https://youtu.be/abc123",
        download_sections=["*0:00-5:00", "*10:00-inf", "*-inf-1:00"],
    )
    ast.parse(code)
    assert "from yt_dlp.utils import download_range_func" in code
    assert "(0.0, 300.0)" in code
    assert "(600.0, float('inf'))" in code
    assert "(float('-inf'), 60.0)" in code
    assert '"force_keyframes_at_cuts": True,' in code
    assert "import re" not in code


def test_transcribe_render_download_sections_chapter_imports_re():
    # A chapter-regex spec compiles to re.compile(...), so the script imports re.
    code = code_gen.transcribe({}, source="https://youtu.be/abc123", download_sections=["intro"])
    ast.parse(code)
    assert "import re" in code
    assert "download_range_func([re.compile('intro')], [], False)" in code


def test_transcribe_render_download_sections_ignored_for_local_file():
    # Sections only apply to the URL download path; a local source generates no yt-dlp code.
    code = code_gen.transcribe({}, source="call.mp3", download_sections=["*0:00-5:00"])
    ast.parse(code)
    assert "download_range_func" not in code
    assert "yt_dlp" not in code
    # No sections in play means no chapter regexes, so no spurious `import re` either.
    assert "import re" not in code


def test_transcribe_render_plain_url_is_not_downloaded():
    # A non-YouTube http(s) URL is uploaded straight through — no yt-dlp scaffolding.
    code = code_gen.transcribe({}, source="https://assembly.ai/wildfires.mp3")
    ast.parse(code)
    assert "yt_dlp" not in code
    assert "tempfile" not in code
    assert "transcriber.transcribe('https://assembly.ai/wildfires.mp3')" in code


# ---------------------------------------------------------------------------
# Validity & fidelity checks (the exhaustive hypothesis harness — Task 10 —
# lives in test_code_gen_fuzz.py).
# ---------------------------------------------------------------------------


def _compiles(code: str) -> None:
    # compile() is stricter than ast.parse() and is what `python file.py` runs through.
    compile(code, "<generated>", "exec")


class _Stub:
    """A transcript-shaped stub exposing every attribute the snippets read."""

    text: ClassVar[str] = "hello world"
    utterances: ClassVar[list[object]] = [type("U", (), {"speaker": "A", "text": "hi"})()]
    summary: ClassVar[str] = "a summary"
    chapters: ClassVar[list[object]] = [type("C", (), {"headline": "intro"})()]
    auto_highlights: ClassVar[object] = type(
        "H", (), {"results": [type("R", (), {"count": 2, "text": "k"})()]}
    )()
    sentiment_analysis: ClassVar[list[object]] = [
        type("S", (), {"sentiment": "POSITIVE", "text": "good"})()
    ]
    entities: ClassVar[list[object]] = [
        type("E", (), {"entity_type": "person_name", "text": "Ada"})()
    ]
    iab_categories: ClassVar[object] = type("I", (), {"summary": {"Tech": 0.9}})()
    content_safety: ClassVar[object] = type("CS", (), {"summary": {"profanity": 0.1}})()


def test_every_snippet_execs_against_a_realistic_transcript() -> None:
    # Enable every feature so result_handling emits all snippets, then exec them.
    all_on: dict[str, object] = {
        "speaker_labels": True,
        "summarization": True,
        "auto_chapters": True,
        "auto_highlights": True,
        "sentiment_analysis": True,
        "entity_detection": True,
        "iab_categories": True,
        "content_safety": True,
    }
    body = snippets.result_handling(all_on)
    exec(compile(body, "<snippets>", "exec"), {"transcript": _Stub()})  # noqa: S102


@pytest.mark.parametrize(
    ("field", "fragment"),
    [
        ("text", "print(transcript.text)"),
        ("id", "print(transcript.id)"),
        ("status", "print(transcript.status.value)"),
        ("utterances", 'print(f"Speaker {utt.speaker}: {utt.text}")'),
        ("srt", "print(transcript.export_subtitles_srt())"),
        ("json", "print(json.dumps(transcript.json_response, default=str))"),
    ],
)
def test_transcribe_render_output_field_generates_matching_code(field, fragment):
    # Each -o choice maps to result code faithful to client._FIELD_RENDERERS.
    code = render_transcribe_code({}, "audio.mp3", output=field)
    _compiles(code)
    assert fragment in code


def test_transcribe_render_output_json_imports_json_only_when_needed():
    assert "import json" in render_transcribe_code({}, "audio.mp3", output="json")
    assert "import json" not in render_transcribe_code({}, "audio.mp3", output="srt")
    assert "import json" not in render_transcribe_code({}, "audio.mp3")


def test_transcribe_render_output_replaces_analysis_result_handling():
    # -o overrides the analysis sections, exactly like the real command's output path.
    code = render_transcribe_code({"speaker_labels": True}, "audio.mp3", output="srt")
    _compiles(code)
    assert "print(transcript.export_subtitles_srt())" in code
    assert "transcript.utterances" not in code


def test_transcribe_render_output_takes_precedence_over_llm_gateway():
    # The real command returns the -o field before the LLM chain runs; the generated
    # script mirrors that and stays free of an unused OpenAI import.
    code = render_transcribe_code(
        {},
        "audio.mp3",
        llm_gateway={"prompts": ["summarize"], "model": "m", "max_tokens": 5},
        output="srt",
    )
    _compiles(code)
    assert "print(transcript.export_subtitles_srt())" in code
    assert "from openai import OpenAI" not in code


def test_transcribe_render_unknown_output_falls_back_to_text():
    # Mirrors select_transcript_field's fallback for unrecognized field names.
    code = render_transcribe_code({}, "audio.mp3", output="bogus")
    _compiles(code)
    assert "print(transcript.text)" in code


def test_transcribe_show_code_includes_llm_gateway_transform():
    code = code_gen.transcribe(
        {"speaker_labels": True},
        "audio.mp3",
        llm_gateway={
            "prompts": ["translate to spanish"],
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
        },
    )
    ast.parse(code)
    assert "from openai import OpenAI" in code
    assert "llm-gateway.assemblyai.com" in code
    assert "translate to spanish" in code
    assert "{{ transcript }}" in code  # gateway injects the transcript at this tag
    assert '"transcript_id": transcript.id' in code
    # The LLM-gateway transform replaces the analysis result-handling (as the CLI does).
    assert "transcript.utterances" not in code


def test_transcribe_show_code_chains_multiple_llm_gateway_prompts():
    code = code_gen.transcribe(
        {},
        "audio.mp3",
        llm_gateway={
            "prompts": ["summarize", "translate the summary to Spanish"],
            "model": "claude-sonnet-4-6",
            "max_tokens": 500,
        },
    )
    ast.parse(code)
    # Both prompts appear, in order, and the script loops to chain them.
    assert "'summarize'," in code
    assert "'translate the summary to Spanish'," in code
    assert "for i, prompt in enumerate(prompts):" in code
    # First step uses the transcript; later steps chain on the previous result,
    # wrapped under the same "Transcript:" label the CLI's run_chain_steps uses.
    assert '"transcript_id": transcript.id' in code
    assert 'content = prompt + "\\n\\nTranscript:\\n" + result' in code


def test_transcribe_show_code_without_gateway_has_no_openai_import():
    code = code_gen.transcribe({"speaker_labels": True}, "audio.mp3")
    assert "from openai import OpenAI" not in code
    assert "transcript.utterances" in code  # normal result handling instead


def test_generated_code_targets_active_environment():
    # --show-code embeds hosts from the active environment, so a sandbox user's
    # generated script talks to the sandbox that minted their key, not production.
    from aai_cli import environments

    sandbox = environments.get("sandbox000")
    environments.set_active(sandbox)

    assert sandbox.streaming_host in code_gen.stream({})
    llm_code = code_gen.stream({}, llm={"prompts": ["p"], "model": "m", "max_tokens": 5})
    assert sandbox.streaming_host in llm_code
    assert sandbox.llm_gateway_base in llm_code
    assert sandbox.agents_host in code_gen.agent("aura", "be brief", "hi")
    transcribe_code = code_gen.transcribe(
        {}, source="a.mp3", llm_gateway={"prompts": ["p"], "model": "m", "max_tokens": 5}
    )
    assert sandbox.llm_gateway_base in transcribe_code
    assert f"aai.settings.base_url = {sandbox.api_base!r}" in transcribe_code


def test_generated_transcribe_omits_base_url_on_production():
    # The SDK already defaults to the production api base, so the default
    # environment's generated script stays free of redundant settings lines.
    assert "base_url" not in code_gen.transcribe({}, source="a.mp3")
