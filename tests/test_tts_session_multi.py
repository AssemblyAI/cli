"""Multi-request streaming-TTS sessions (aai_cli/tts/session.py): long text sent as
bounded Generate + Flush requests on one socket, unknown frames skipped, and the
documented Terminate -> Termination end of session. Split from test_tts_session.py
to keep both under the 500-line file gate."""

from __future__ import annotations

import json

import pytest

from aai_cli.core.errors import APIError
from aai_cli.tts import session
from tests._tts_session_helpers import (
    FakeWS,
    audio_chunk,
    begin_frame,
    connect_returning,
    flush_done_frame,
)


def test_synthesize_chunked_sends_every_chunk_as_a_request_on_one_session(monkeypatch):
    # The documented multi-request pattern: one socket, a Generate + Flush per chunk,
    # never the whole document in one Generate nor a socket per chunk (3009).
    monkeypatch.setattr("aai_cli.tts.text.chunk_text", lambda _t: ["First chunk.", "Second chunk."])
    ws = FakeWS(
        [
            begin_frame(),
            audio_chunk(b"\x01\x02"),
            flush_done_frame(),
            audio_chunk(b"\x03\x04"),
            flush_done_frame(),
        ]
    )
    opened: list[str] = []
    result = session.synthesize_chunked(
        "k",
        session.SpeakConfig(text="ignored"),
        connect=lambda url, **k: opened.append(url) or ws,
    )
    assert len(opened) == 1
    assert ws.sent == [
        {"type": "Generate", "text": "First chunk."},
        {"type": "Flush"},
        {"type": "Generate", "text": "Second chunk."},
        {"type": "Flush"},
        {"type": "Terminate"},
    ]
    # PCM is concatenated across the requests, and the duration recomputed from it.
    assert result.pcm == b"\x01\x02\x03\x04"
    assert result.audio_duration_seconds == pytest.approx(4 / 2 / 24000)


def test_synthesize_bounds_requests_in_flight(monkeypatch):
    # At most three requests go out ahead of their FlushDone; the fourth waits for the
    # first FlushDone, so a long document never arrives as one burst (3010).
    monkeypatch.setattr("aai_cli.tts.text.chunk_text", lambda _t: ["a.", "b.", "c.", "d."])
    sent_before_first_done: list[int] = []
    done_before_terminate: list[int] = []

    class _Watching(FakeWS):
        done = 0

        def recv(self, timeout: float | None = None) -> str:
            frame = super().recv(timeout)
            if json.loads(frame)["type"] == "FlushDone":
                if not self.done:
                    sent_before_first_done.append(len(self.sent))
                self.done += 1
            return frame

        def send(self, data: str) -> None:
            if json.loads(data)["type"] == "Terminate":
                done_before_terminate.append(self.done)
            super().send(data)

    ws = _Watching([begin_frame(), *[flush_done_frame()] * 4])
    session.synthesize_chunked("k", session.SpeakConfig(text="x"), connect=lambda *a, **k: ws)
    assert sent_before_first_done == [6]  # three Generate + Flush pairs
    # Terminate waits until every request's FlushDone has arrived.
    assert done_before_terminate == [4]
    assert [m.get("text") for m in ws.sent if m["type"] == "Generate"] == ["a.", "b.", "c.", "d."]


def test_synthesize_chunked_streams_each_chunk_to_on_audio(monkeypatch):
    # on_audio fires per chunk as it arrives (incremental playback), carrying the
    # session's reported sample rate, across all requests.
    monkeypatch.setattr("aai_cli.tts.text.chunk_text", lambda _t: ["a.", "b."])
    ws = FakeWS(
        [
            begin_frame(sample_rate=16000),
            audio_chunk(b"\x01\x02"),
            flush_done_frame(),
            audio_chunk(b"\x03\x04"),
            flush_done_frame(),
        ]
    )
    streamed: list[tuple[bytes, int]] = []
    session.synthesize_chunked(
        "k",
        session.SpeakConfig(text="x"),
        connect=lambda *a, **k: ws,
        on_audio=lambda chunk, rate: streamed.append((chunk, rate)),
    )
    assert streamed == [(b"\x01\x02", 16000), (b"\x03\x04", 16000)]


def test_synthesize_ignores_unknown_frames():
    # The protocol may add frame types (WordBoundaries today); they must be skipped.
    ws = FakeWS(
        [
            begin_frame(),
            audio_chunk(b"\x01\x02"),
            flush_done_frame(),
            json.dumps({"type": "WordBoundaries", "flush_id": 0, "words": []}),
        ]
    )
    result = session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert result.pcm == b"\x01\x02"


def test_terminate_waits_for_termination_skipping_late_frames():
    # After Terminate, frames like a late WordBoundaries arrive before Termination.
    ws = FakeWS(
        [
            begin_frame(),
            flush_done_frame(),
            json.dumps({"type": "WordBoundaries", "flush_id": 0, "words": []}),
            json.dumps({"type": "Termination"}),
        ]
    )
    session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)
    assert ws.recv_timeouts == [60.0, 60.0, 60.0, 60.0]


def test_error_after_terminate_is_an_api_error():
    ws = FakeWS(
        [
            begin_frame(),
            flush_done_frame(),
            json.dumps({"type": "Error", "error_code": 3005, "error": "at capacity"}),
        ]
    )
    with pytest.raises(APIError, match=r"TTS error \(3005\): at capacity"):
        session.synthesize("k", session.SpeakConfig(text="hi"), connect=lambda *a, **k: ws)


def test_synthesize_chunked_single_chunk_passes_text_through():
    # A short input is one chunk: the original text is synthesized unchanged.
    captured: dict = {}
    ws = FakeWS([begin_frame(), audio_chunk(b"\x01\x02"), flush_done_frame()])
    session.synthesize_chunked(
        "k", session.SpeakConfig(text="Just one sentence."), connect=connect_returning(ws, captured)
    )
    assert ws.sent[0]["text"] == "Just one sentence."


def test_synthesize_chunked_empty_text_returns_silent_default():
    # Whitespace-only text yields no chunks -> no audio at the default rate, and no crash.
    # connect is omitted entirely: the loop never runs, so no connection is attempted.
    result = session.synthesize_chunked("k", session.SpeakConfig(text="   "))
    assert result.pcm == b""
    assert result.sample_rate == 24000
    assert result.audio_duration_seconds == 0.0
