from __future__ import annotations

import pytest

from aai_cli.tts import session
from tests._tts_session_helpers import FakeWS, audio_frame, begin_frame


def test_synthesize_dialogue_concatenates_segments_with_silence():
    # One fresh fake socket per segment; record the voice each connection requested.
    sockets = [
        FakeWS([begin_frame(sample_rate=24000), audio_frame(b"\xaa\xbb", final=True)]),
        FakeWS([begin_frame(sample_rate=24000), audio_frame(b"\xcc\xdd", final=True)]),
    ]
    urls: list[str] = []

    def _connect(url: str, **_kwargs):
        urls.append(url)
        return sockets.pop(0)

    result = session.synthesize_dialogue(
        "k",
        [("jane", "Hello."), ("michael", "Hi.")],
        language="English",
        connect=_connect,
    )
    # Each segment connected with its own voice.
    assert "voice=jane" in urls[0]
    assert "voice=michael" in urls[1]
    # 0.25 s of silence (24000 * 0.25 * 2 = 12000 zero bytes) sits BETWEEN the two
    # segments' PCM, with none at the ends.
    gap = b"\x00" * 12000
    assert result.pcm == b"\xaa\xbb" + gap + b"\xcc\xdd"
    assert result.sample_rate == 24000
    # Pin the duration formula (len/2/rate) so its operators survive the mutation gate.
    assert result.audio_duration_seconds == pytest.approx(len(result.pcm) / 2 / 24000)


def test_synthesize_dialogue_single_segment_has_no_silence():
    ws = FakeWS([begin_frame(sample_rate=24000), audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize_dialogue("k", [("jane", "Hi.")], connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"  # no leading/trailing pad


def test_synthesize_dialogue_uses_server_sample_rate():
    # A non-default server rate must flow into the result (proving the per-segment
    # rate is read, not left at the default) and into the duration denominator.
    ws = FakeWS([begin_frame(sample_rate=16000), audio_frame(b"\x01\x02", final=True)])
    result = session.synthesize_dialogue("k", [("jane", "Hi.")], connect=lambda *a, **k: ws)
    assert result.sample_rate == 16000
    assert result.audio_duration_seconds == pytest.approx(2 / 2 / 16000)


def test_synthesize_dialogue_empty_segments_returns_silent_default():
    # No segments -> no audio at the default rate, and no crash. connect is omitted
    # entirely: the loop body never runs, so no connection is ever attempted.
    result = session.synthesize_dialogue("k", [])
    assert result.pcm == b""
    assert result.sample_rate == 24000
    assert result.audio_duration_seconds == 0.0
