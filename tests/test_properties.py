"""Property-based tests for the encoding/parsing-heavy paths."""

import io
import json
import math
import types
import wave
from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aai_cli import config_builder as cb
from aai_cli import jsonshape, timeparse, wer
from aai_cli.agent.render import AgentRenderer
from aai_cli.errors import UsageError
from aai_cli.streaming import sources
from aai_cli.streaming.render import StreamRenderer


@given(text=st.text())
def test_agent_json_preserves_arbitrary_text(text):
    # Quotes, newlines, unicode, control chars must survive the NDJSON round-trip.
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).user_final(text)
    events = [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]
    assert {"type": "transcript.user", "text": text} in events


@given(text=st.text())
def test_stream_json_preserves_arbitrary_transcript(text):
    buf = io.StringIO()
    StreamRenderer(json_mode=True, out=buf).turn(
        types.SimpleNamespace(transcript=text, end_of_turn=True)
    )
    assert json.loads(buf.getvalue()) == {
        "type": "turn",
        "transcript": text,
        "end_of_turn": True,
    }


@given(pcm=st.binary(max_size=8000))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
def test_wav_chunks_reassemble_and_stay_bounded(pcm, tmp_path):
    pcm = pcm[: len(pcm) // 2 * 2]  # whole 16-bit mono frames
    assume(pcm)  # the empty-file case is covered by a dedicated unit test
    clip = tmp_path / "clip.wav"
    with wave.open(str(clip), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sources.TARGET_RATE)
        w.writeframes(pcm)
    chunks = list(sources.FileSource(str(clip), sleep=lambda _s: None))
    assert b"".join(chunks) == pcm  # streamed audio is byte-exact
    assert all(len(c) <= sources.CHUNK_BYTES for c in chunks)  # chunking respects the cap


# --- config-builder coercion round-trips ----------------------------------


@given(value=st.integers(min_value=0, max_value=10_000_000))
def test_int_coercion_roundtrips(value):
    assert cb.coerce_value("speakers_expected", str(value)) == value


@given(value=st.lists(st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=6), max_size=5))
def test_list_coercion_roundtrips(value):
    raw = ",".join(value)
    assert cb.coerce_value("word_boost", raw) == [v for v in value if v]


@given(value=st.booleans())
def test_bool_coercion_roundtrips(value):
    assert cb.coerce_value("speaker_labels", str(value).lower()) is value


@given(value=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_float_coercion_roundtrips(value):
    assert cb.coerce_value("speech_threshold", repr(value)) == value


# --- config-builder coercion: arbitrary (mostly bad) input -----------------
#
# Oracle-style properties: the coercer must agree with the underlying Python
# parser on EVERY string — accept exactly what it accepts, and turn every
# rejection into a UsageError naming the field (never a raw traceback).


@given(raw=st.text(max_size=16))
def test_bool_coercion_accepts_exactly_the_literals(raw):
    low = raw.strip().lower()
    if low in {"1", "true", "yes", "on"}:
        assert cb.coerce_value("speaker_labels", raw) is True
    elif low in {"0", "false", "no", "off"}:
        assert cb.coerce_value("speaker_labels", raw) is False
    else:
        with pytest.raises(UsageError, match="speaker_labels"):
            cb.coerce_value("speaker_labels", raw)


@given(raw=st.text(max_size=16))
def test_int_coercion_mirrors_python_int(raw):
    try:
        expected = int(raw)
    except ValueError:
        with pytest.raises(UsageError, match="speakers_expected"):
            cb.coerce_value("speakers_expected", raw)
    else:
        assert cb.coerce_value("speakers_expected", raw) == expected


@given(raw=st.text(max_size=16))
def test_float_coercion_mirrors_python_float(raw):
    try:
        expected = float(raw)
    except ValueError:
        with pytest.raises(UsageError, match="speech_threshold"):
            cb.coerce_value("speech_threshold", raw)
    else:
        got = cb.coerce_value("speech_threshold", raw)
        assert got == expected or (
            isinstance(got, float) and math.isnan(got) and math.isnan(expected)
        )


@given(raw=st.text(max_size=24))
def test_json_coercion_mirrors_json_loads(raw):
    try:
        expected: object = json.loads(raw)
    except json.JSONDecodeError:
        with pytest.raises(UsageError, match="custom_spelling"):
            cb.coerce_value("custom_spelling", raw)
    else:
        got = cb.coerce_value("custom_spelling", raw)
        assert got == expected or (
            isinstance(got, float) and math.isnan(got) and isinstance(expected, float)
        )


@given(pair=st.text(max_size=20))
def test_config_pair_without_equals_is_always_a_usage_error(pair):
    assume("=" not in pair)
    with pytest.raises(UsageError, match="KEY=VALUE"):
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, [pair])


@given(key=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=24))
def test_unknown_config_field_is_always_a_usage_error(key):
    assume(key not in cb.TRANSCRIBE_FIELDS)
    with pytest.raises(UsageError, match="Unknown config field"):
        cb.parse_config_overrides(cb.TRANSCRIBE_FIELDS, [f"{key}=x"])


# --- timeparse: UTC normalization ------------------------------------------

_offsets = st.integers(min_value=-23 * 60, max_value=23 * 60).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
_datetimes = st.datetimes(min_value=datetime(1970, 1, 2), max_value=datetime(2099, 12, 30))
_aware = st.builds(lambda dt, tz: dt.replace(tzinfo=tz), _datetimes, _offsets)


@given(dt=_aware)
def test_parse_iso_utc_normalizes_any_offset_to_utc(dt):
    parsed = timeparse.parse_iso_utc(dt.isoformat())
    assert parsed == dt  # same instant
    assert parsed is not None and parsed.tzinfo is UTC  # rendered in UTC


@given(dt=_datetimes)
def test_parse_iso_utc_treats_naive_and_z_suffixed_as_utc(dt):
    as_utc = dt.replace(tzinfo=UTC)
    assert timeparse.parse_iso_utc(dt.isoformat()) == as_utc
    assert timeparse.parse_iso_utc(dt.isoformat() + "Z") == as_utc


@given(dt=_aware)
def test_format_utc_day_matches_the_datetime_render(dt):
    iso = dt.isoformat()
    rendered = timeparse.format_utc_datetime(iso)
    assert rendered.startswith(timeparse.format_utc_day(iso))
    assert datetime.strptime(rendered, "%Y-%m-%d %H:%M:%S") is not None


@given(value=st.text(max_size=24))
def test_unparseable_text_falls_back_verbatim(value):
    assume(timeparse.parse_iso_utc(value) is None)
    assert timeparse.format_utc_day(value) == (value or "")
    assert timeparse.format_utc_datetime(value) == (value or "")


@given(
    value=st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.lists(st.integers(), max_size=3)
)
def test_parse_iso_utc_rejects_every_non_string(value):
    assert timeparse.parse_iso_utc(value) is None


# --- jsonshape: total functions over arbitrary JSON -------------------------

_jsonish = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=8),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=4), children, max_size=3)
    ),
    max_leaves=8,
)


@given(value=_jsonish)
def test_as_int_is_total_and_bools_are_not_counts(value):
    result = jsonshape.as_int(value, default=-99)
    if isinstance(value, bool):
        assert result == -99
    elif isinstance(value, int):
        assert result == value


@given(value=_jsonish)
def test_as_float_is_total_and_bools_are_not_counts(value):
    result = jsonshape.as_float(value, default=-99.0)
    if isinstance(value, bool):
        assert result == -99.0
    elif isinstance(value, int | float):
        assert result == float(value)


@given(value=_jsonish)
def test_mapping_list_keeps_exactly_the_objects(value):
    expected = [
        item for item in (value if isinstance(value, list) else []) if isinstance(item, dict)
    ]
    assert jsonshape.mapping_list(value) == expected


@given(value=_jsonish)
def test_as_object_list_is_all_or_nothing(value):
    result = jsonshape.as_object_list(value)
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        assert result == value
    else:
        assert result is None


# --- WER metric invariants ---------------------------------------------------

_word = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8)
_sentence = st.lists(_word, min_size=1, max_size=12).map(" ".join)


@given(text=_sentence)
@settings(deadline=None)  # first example pays jiwer's lazy import
def test_wer_identity_scores_zero(text):
    n = len(text.split())  # every word survives normalization (lowercase a-z)
    assert wer.score(text, text) == wer.Score(errors=0, words=n)


@given(text=_sentence)
@settings(deadline=None)
def test_wer_against_empty_hypothesis_is_one(text):
    score = wer.score(text, "")
    assert score.words == len(text.split())
    assert score.errors == score.words  # every reference word is a deletion
    assert score.wer == 1.0


@given(scores=st.lists(st.builds(wer.Score, st.integers(0, 100), st.integers(1, 100)), max_size=8))
def test_pooled_wer_sums_errors_and_words(scores):
    combined = wer.pooled(scores)
    assert combined.errors == sum(s.errors for s in scores)
    assert combined.words == sum(s.words for s in scores)
