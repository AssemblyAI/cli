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
