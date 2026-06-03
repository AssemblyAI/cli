from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from assemblyai_cli.code_gen import serialize


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

from assemblyai_cli import config_builder  # noqa: E402

# JSON-ish values that repr()->eval() round-trips (string keys, no NaN/inf).
_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(st.characters(blacklist_categories=["Cs"]), max_size=8),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(
        st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5),
        children,
        max_size=3,
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


from assemblyai_cli.code_gen import snippets  # noqa: E402


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

    from assemblyai_cli import transcribe_render

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

from assemblyai_cli import code_gen  # noqa: E402


def test_transcribe_render_parses_and_uses_env_key():
    code = code_gen.transcribe({"speaker_labels": True}, source="https://assembly.ai/wildfires.mp3")
    ast.parse(code)  # raises SyntaxError if malformed
    assert 'os.environ["ASSEMBLYAI_API_KEY"]' in code
    assert "https://assembly.ai/wildfires.mp3" in code
    assert "transcript.utterances" in code  # result handling for speaker_labels
    assert "{{API_KEY}}" not in code  # never echo a real key


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
    code = code_gen.agent(voice="ivy", system_prompt='Say "hi"', greeting="Hello")
    ast.parse(code)  # must stay valid Python despite the embedded quotes
