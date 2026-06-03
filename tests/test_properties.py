"""Property-based tests for the encoding/parsing-heavy paths."""

import io
import json
import types
import wave

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from assemblyai_cli import config_builder as cb
from assemblyai_cli.agent.render import AgentRenderer
from assemblyai_cli.streaming import sources
from assemblyai_cli.streaming.render import StreamRenderer


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
