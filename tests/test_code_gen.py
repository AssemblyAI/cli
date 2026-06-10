from __future__ import annotations

from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aai_cli.code_gen import serialize
from aai_cli.code_gen.transcribe import render as render_transcribe_code

settings.register_profile("codegen", max_examples=150)
settings.load_profile("codegen")


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


# ---------------------------------------------------------------------------
# Shared, domain-driven strategy: build merged-kwargs dicts from the AUTHORITATIVE
# field tables in config_builder. Used by every validity test below. Because the
# field list comes from the coerce tables, any field added later is fuzzed for free.
# ---------------------------------------------------------------------------
from assemblyai.streaming.v3 import SpeechModel  # noqa: E402

from aai_cli import config_builder  # noqa: E402

# JSON-ish values that repr()->eval() round-trips (string keys, no NaN/inf).
_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(st.characters(blacklist_categories=["Cs"]), max_size=8),
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(
            st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5),
            children,
            max_size=3,
        )
    ),
    max_leaves=5,
)

_BY_KIND = {
    "str": st.text(st.characters(blacklist_categories=["Cs"]), max_size=16),
    "bool": st.booleans(),
    "int": st.integers(),
    "float": st.floats(allow_nan=False, allow_infinity=False),
    "list": st.lists(st.text(st.characters(blacklist_categories=["Cs"]), max_size=8), max_size=4),
    "json": _json,
}


def _value_for(field: str, kind: str):
    # speech_model in the streaming table may be a SpeechModel enum in real merged dicts.
    if field == "speech_model":
        return st.sampled_from(list(SpeechModel)) | _BY_KIND["str"]
    return _BY_KIND[kind]


def merged_strategy(coerce_table: dict[str, str]) -> st.SearchStrategy:
    """A hypothesis strategy yielding merged-kwargs dicts over the FULL field table."""
    return st.fixed_dictionaries(
        {}, optional={f: _value_for(f, kind) for f, kind in coerce_table.items()}
    )


@given(merged_strategy(config_builder.TRANSCRIBE_COERCE))
def test_serializer_round_trips_full_transcribe_domain(merged):
    lines = serialize.config_kwarg_lines(merged, indent=0)
    src = "dict(\n" + "\n".join(lines) + "\n)"
    assert eval(src, {"SpeechModel": SpeechModel}) == merged  # noqa: S307


@given(merged_strategy(config_builder.STREAM_COERCE))
def test_serializer_round_trips_full_stream_domain(merged):
    lines = serialize.config_kwarg_lines(merged, indent=0)
    src = "dict(\n" + "\n".join(lines) + "\n)"
    assert eval(src, {"SpeechModel": SpeechModel}) == merged  # noqa: S307


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


def test_stream_render_parses_and_is_runnable_shape():
    from assemblyai.streaming.v3 import SpeechModel

    code = code_gen.stream(
        {"sample_rate": 16000, "format_turns": True, "speech_model": SpeechModel.u3_rt_pro}
    )
    ast.parse(code)
    assert "StreamingClient(" in code
    assert "StreamingParameters(" in code
    assert "SpeechModel.u3_rt_pro" in code
    assert "MicrophoneStream" in code
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code


def test_stream_render_mic_rate_matches_params():
    code = code_gen.stream({"sample_rate": 8000})
    ast.parse(code)
    assert "StreamingParameters(\n        sample_rate=8000," in code
    assert "MicrophoneStream(sample_rate=8000)" in code


def test_stream_render_empty_is_clean_and_has_no_speechmodel_import():
    code = code_gen.stream({})
    ast.parse(code)
    assert "StreamingParameters()" in code
    assert "    SpeechModel," not in code  # not imported when unused (keeps script lint-clean)
    assert "MicrophoneStream(sample_rate=16000)" in code  # default rate


def test_agent_render_parses_and_injects_session_fields():
    code = code_gen.agent(voice="ivy", system_prompt="Be terse.", greeting="Hi there")
    ast.parse(code)
    assert '"voice": "ivy"' in code
    assert "Be terse." in code
    assert "Hi there" in code
    assert "agents.assemblyai.com" in code
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code


def test_agent_render_escapes_quotes_in_prompt():
    import json as _json

    tricky = 'Say "hi"\nand stop'
    code = code_gen.agent(voice="ivy", system_prompt=tricky, greeting="Hello")
    ast.parse(code)  # valid Python despite embedded quotes/newlines
    # The prompt is injected via json.dumps, so its escaped form appears verbatim.
    assert _json.dumps(tricky) in code


# ---------------------------------------------------------------------------
# Exhaustive validity & fidelity harness (Task 10)
# ---------------------------------------------------------------------------


def _compiles(code: str) -> None:
    # compile() is stricter than ast.parse() and is what `python file.py` runs through.
    compile(code, "<generated>", "exec")


@given(merged_strategy(config_builder.TRANSCRIBE_COERCE))
def test_fuzz_transcribe_always_compiles(merged):
    _compiles(code_gen.transcribe(merged, source="audio.mp3"))


@given(merged_strategy(config_builder.STREAM_COERCE))
def test_fuzz_stream_always_compiles(merged):
    _compiles(code_gen.stream(merged))


@given(
    voice=st.text(st.characters(blacklist_categories=["Cs"]), max_size=20),
    system_prompt=st.text(st.characters(blacklist_categories=["Cs"]), max_size=200),
    greeting=st.text(st.characters(blacklist_categories=["Cs"]), max_size=200),
)
def test_fuzz_agent_always_compiles(voice, system_prompt, greeting):
    # Arbitrary text (quotes, newlines, backslashes, unicode) must never break the script.
    _compiles(code_gen.agent(voice=voice, system_prompt=system_prompt, greeting=greeting))


@given(merged_strategy(config_builder.TRANSCRIBE_COERCE))
def test_fuzz_transcribe_config_round_trips_in_generated_code(merged):
    # The TranscriptionConfig(...) the generated code builds must equal the merged dict.
    code = code_gen.transcribe(merged, source="audio.mp3")
    if not merged:
        assert "TranscriptionConfig(" not in code
        return
    # repr() escapes newlines, so no kwarg line contains a literal "\n)"; the first
    # "\n)" after the constructor opens is always the config block's closer.
    inner = code.split("aai.TranscriptionConfig(\n", 1)[1].split("\n)", 1)[0]
    rebuilt = eval("dict(\n" + inner + "\n)", {"SpeechModel": SpeechModel})  # noqa: S307
    assert rebuilt == merged


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


@given(merged_strategy(config_builder.TRANSCRIBE_COERCE))
def test_fuzz_result_handling_always_execs(merged):
    body = snippets.result_handling(merged)
    exec(compile(body, "<snippets>", "exec"), {"transcript": _Stub(), "getattr": getattr})  # noqa: S102


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


@given(
    merged=merged_strategy(config_builder.TRANSCRIBE_COERCE),
    field=st.sampled_from(["text", "id", "status", "utterances", "srt", "json"]),
)
def test_fuzz_transcribe_output_fields_always_compile(merged, field):
    _compiles(render_transcribe_code(merged, "audio.mp3", output=field))


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


def test_agent_show_code_uses_single_full_duplex_stream():
    # ONE sd.RawStream (mic+speaker); two separate streams fail on macOS CoreAudio.
    code = code_gen.agent(voice="ivy", system_prompt="p", greeting="g")
    ast.parse(code)
    assert "sd.RawStream(" in code
    assert "samplerate=RATE" in code  # opens at the agent's native 24 kHz, no resampling
    assert "RawInputStream" not in code
    assert "RawOutputStream" not in code
    # No audioop: it's deprecated and removed in Python 3.13, so the script stays portable.
    assert "audioop" not in code


def test_stream_show_code_includes_llm_follow_loop():
    code = code_gen.stream(
        {"speech_model": "universal_streaming"},
        llm={
            "prompts": ["summarize", "translate to french"],
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 500,
            "interval": 30.0,
        },
    )
    ast.parse(code)
    assert "from openai import OpenAI" in code
    assert "llm-gateway.assemblyai.com" in code
    # Both prompts appear, in order, for the chain.
    assert code.index("summarize") < code.index("translate to french")
    # Still streams from the mic, refreshing the answer on the interval.
    assert "MicrophoneStream" in code
    assert "end_of_turn" in code
    assert "claude-haiku-4-5-20251001" in code
    # The generated loop mirrors --llm-interval: a baked-in throttle plus a closing flush.
    assert "LLM_INTERVAL = 30.0" in code
    assert "now - _last_summary < LLM_INTERVAL" in code
    assert "summarize(final=True)" in code


def test_gateway_options_defaults_interval_to_per_turn():
    # Called without an explicit interval (transcribe's path), the baked-in cadence is
    # per-turn (0.0); pins the default so it can't drift.
    opts = code_gen.gateway_options(["summarize"], "m", 100)
    assert opts is not None
    assert opts["interval"] == 0.0


def test_stream_show_code_defaults_interval_when_absent():
    # An llm dict with no "interval" key falls back to per-turn (LLM_INTERVAL = 0.0).
    code = code_gen.stream({}, llm={"prompts": ["s"], "model": "m", "max_tokens": 1})
    ast.parse(code)
    assert "LLM_INTERVAL = 0.0" in code


def test_stream_show_code_llm_interval_zero_is_per_turn():
    # --llm-interval 0 bakes in the legacy per-turn cadence (LLM_INTERVAL = 0.0 makes the
    # throttle a no-op), while still emitting the closing flush.
    code = code_gen.stream(
        {},
        llm=code_gen.gateway_options(["summarize"], "m", 100, interval=0.0),
    )
    ast.parse(code)
    assert "LLM_INTERVAL = 0.0" in code
    assert "summarize(final=True)" in code


def test_stream_show_code_without_llm_is_plain_scaffold():
    code = code_gen.stream({})
    ast.parse(code)
    assert "from openai import OpenAI" not in code  # no gateway when --llm absent
    assert "MicrophoneStream" in code


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
