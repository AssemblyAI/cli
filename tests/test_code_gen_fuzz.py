"""Hypothesis fuzz/property tests for code_gen: validity and round-trip fidelity.

Example-based code_gen tests live in test_code_gen.py.
"""

from __future__ import annotations

from typing import ClassVar

from assemblyai.streaming.v3 import SpeechModel
from hypothesis import given, settings
from hypothesis import strategies as st

from aai_cli import code_gen, config_builder
from aai_cli.code_gen import serialize, snippets
from aai_cli.code_gen.transcribe import render as render_transcribe_code

settings.register_profile("codegen", max_examples=150)
settings.load_profile("codegen")

# ---------------------------------------------------------------------------
# Shared, domain-driven strategy: build merged-kwargs dicts from the AUTHORITATIVE
# field tables in config_builder. Used by every validity test below. Because the
# field list comes from the coerce tables, any field added later is fuzzed for free.
# ---------------------------------------------------------------------------

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


@given(merged_strategy(config_builder.TRANSCRIBE_COERCE))
def test_fuzz_result_handling_always_execs(merged):
    body = snippets.result_handling(merged)
    exec(compile(body, "<snippets>", "exec"), {"transcript": _Stub(), "getattr": getattr})  # noqa: S102


@given(
    merged=merged_strategy(config_builder.TRANSCRIBE_COERCE),
    field=st.sampled_from(["text", "id", "status", "utterances", "srt", "vtt", "json"]),
    chars_per_caption=st.one_of(st.none(), st.integers(min_value=1, max_value=500)),
)
def test_fuzz_transcribe_output_fields_always_compile(merged, field, chars_per_caption):
    _compiles(
        render_transcribe_code(
            merged, "audio.mp3", output=field, chars_per_caption=chars_per_caption
        )
    )
