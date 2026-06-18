"""Tests for the `assembly code` voice I/O (code_agent/voice.py + _exec voice helpers).

The bodies are intentionally unannotated: they drive the voice session through
lightweight fakes (a fake mic, stream_fn, synth_fn, and player) so no microphone,
speaker, or socket is ever touched — the strict type-checker skips untyped test bodies.
"""

from __future__ import annotations

from types import SimpleNamespace

from aai_cli.code_agent import voice as voicemod
from aai_cli.code_agent.voice import VoiceSession, build_voice_session, spoken_summary


class FakeMic:
    def __init__(self, chunks, sample_rate=16000):
        self._chunks = list(chunks)
        self.sample_rate = sample_rate

    def __iter__(self):
        return iter(self._chunks)


def _turn(text, *, end_of_turn):
    return SimpleNamespace(transcript=text, end_of_turn=end_of_turn)


def test_listen_returns_final_turn_and_gates_mic_after_it():
    seen = {}

    def fake_stream(api_key, source, *, params, on_turn):
        seen["key"] = api_key
        seen["params"] = params
        it = iter(source)
        seen["before"] = next(it)  # the first chunk flows before the turn finalizes
        on_turn(_turn("add a verbose flag", end_of_turn=True))
        seen["after"] = list(it)  # gated() must stop now, yielding nothing more

    session = VoiceSession(
        api_key="k",
        readback=False,
        mic_factory=lambda: FakeMic([b"a", b"b", b"c"]),
        stream_fn=fake_stream,
    )
    assert session.listen() == "add a verbose flag"
    assert seen["key"] == "k"
    assert seen["before"] == b"a"
    assert seen["after"] == []  # the mic was gated shut the instant the turn finalized
    assert seen["params"].format_turns is True
    assert seen["params"].sample_rate == 16000


def test_listen_stops_capturing_when_cancelled():
    seen = {}
    holder = {}

    def fake_stream(api_key, source, *, params, on_turn):
        it = iter(source)
        seen["first"] = next(it)  # one chunk flows before the interrupt
        holder["session"].cancel()  # the TUI's Ctrl-C, from another thread
        seen["rest"] = list(it)  # gated() must stop the instant cancel() fires

    session = VoiceSession(
        api_key="k",
        readback=False,
        mic_factory=lambda: FakeMic([b"a", b"b", b"c"]),
        stream_fn=fake_stream,
    )
    holder["session"] = session
    assert session.listen() is None  # cancelled mid-capture -> no turn finalized
    assert seen["first"] == b"a"
    assert seen["rest"] == []  # the mic was gated shut by cancel(), not drained


def test_listen_clears_a_stale_cancel_before_capturing():
    # A cancel() that fired outside a capture must not preempt the next listen — listen()
    # clears the flag on entry, so the gate is open and the turn is captured normally.
    def fake_stream(api_key, source, *, params, on_turn):
        it = iter(source)
        next(it)  # if the stale cancel weren't cleared, gated() would yield nothing here
        on_turn(_turn("hello", end_of_turn=True))
        list(it)

    session = VoiceSession(
        api_key="k",
        readback=False,
        mic_factory=lambda: FakeMic([b"a", b"b"]),
        stream_fn=fake_stream,
    )
    session.cancel()  # a stale cancel set before the capture begins
    assert session.listen() == "hello"  # cleared on entry -> capture proceeds


def test_listen_ignores_partials_and_returns_none_without_a_final_turn():
    def fake_stream(api_key, source, *, params, on_turn):
        on_turn(_turn("typing in progr", end_of_turn=False))  # interim only
        on_turn(_turn("", end_of_turn=True))  # finalized but empty -> not captured
        on_turn(SimpleNamespace(transcript="no end_of_turn field"))  # missing attr -> not final
        list(source)

    session = VoiceSession(
        api_key="k", readback=False, mic_factory=lambda: FakeMic([b"a"]), stream_fn=fake_stream
    )
    # A turn is captured only when end_of_turn is truthy; a partial, an empty final, and an
    # event lacking the field entirely (the getattr default is False) all leave it None.
    assert session.listen() is None


class FakePlayer:
    def __init__(self):
        self.fed = []
        self.exit_exc_type = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *exc):
        self.exit_exc_type = exc_type  # records the abort path (an exception on the way out)
        return False

    def feed(self, pcm, sample_rate):
        self.fed.append((pcm, sample_rate))


def test_speak_synthesizes_and_plays_when_readback_on():
    player = FakePlayer()
    captured = {}

    def fake_synth(api_key, config, *, on_audio):
        captured["text"] = config.text
        captured["rate"] = config.sample_rate
        on_audio(b"pcm", 24000)
        return SimpleNamespace(pcm=b"pcm", sample_rate=24000, audio_duration_seconds=0.0)

    session = VoiceSession(
        api_key="k", readback=True, synth_fn=fake_synth, player_factory=lambda: player
    )
    session.speak("  hello there  ")
    assert captured["text"] == "hello there"  # stripped
    assert captured["rate"] == 24000
    assert player.fed == [(b"pcm", 24000)]


def test_speak_stops_synthesis_and_aborts_player_when_cancelled():
    player = FakePlayer()
    holder = {}
    reached_after_cancel = []

    def fake_synth(api_key, config, *, on_audio):
        on_audio(b"one", 24000)  # first chunk plays
        holder["session"].cancel()  # the user interrupts the readback
        on_audio(b"two", 24000)  # the feed must raise here, ending synthesis
        reached_after_cancel.append(True)  # so this line is never reached

    session = VoiceSession(
        api_key="k", readback=True, synth_fn=fake_synth, player_factory=lambda: player
    )
    holder["session"] = session
    session.speak("hello there")  # returns cleanly — the cancel sentinel is swallowed
    assert player.fed == [(b"one", 24000)]  # only the pre-cancel chunk played
    assert reached_after_cancel == []  # synthesis stopped at the cancelled feed
    assert player.exit_exc_type is not None  # player saw the exception -> aborted, not drained


def test_speak_clears_a_stale_cancel_before_playing():
    # A cancel() left set from a prior interrupt must not abort the next readback before it
    # starts — speak() clears the flag on entry, so the chunk plays normally.
    player = FakePlayer()

    def fake_synth(api_key, config, *, on_audio):
        on_audio(b"pcm", 24000)

    session = VoiceSession(
        api_key="k", readback=True, synth_fn=fake_synth, player_factory=lambda: player
    )
    session.cancel()  # a stale cancel set before this readback
    session.speak("hello")
    assert player.fed == [(b"pcm", 24000)]  # cleared on entry -> the chunk still played


def test_speak_is_a_noop_when_readback_off_or_text_blank():
    def boom(*a, **k):
        raise AssertionError("synthesize must not be called")

    off = VoiceSession(api_key="k", readback=False, synth_fn=boom, player_factory=FakePlayer)
    off.speak("hi")  # readback off -> no synthesis

    blank = VoiceSession(api_key="k", readback=True, synth_fn=boom, player_factory=FakePlayer)
    blank.speak("   ")  # blank text -> no synthesis


def test_spoken_summary_strips_code_and_keeps_prose():
    text = (
        "Here's the fix.\n\n```python\ndef f():\n    return 1\n```\n\n"
        "Call it with `f()` when ready."
    )
    summary = spoken_summary(text)
    # The fenced block and the inline `f()` are gone; only the prose is read aloud.
    assert "def f" not in summary and "return 1" not in summary
    assert "`" not in summary
    assert summary == "Here's the fix. Call it with when ready."


def test_spoken_summary_falls_back_when_reply_is_all_code():
    # A reply that is nothing but a code block leaves no prose -> a generic spoken note,
    # never an empty utterance.
    assert spoken_summary("```\nprint('hi')\n```") == voicemod._ALL_CODE_READBACK


def test_spoken_summary_truncates_long_prose():
    long_prose = "word " * 400  # far over the cap
    summary = spoken_summary(long_prose)
    assert summary.endswith("…")
    assert len(summary) <= voicemod._MAX_SPOKEN_CHARS + 1  # capped prose plus the ellipsis


def test_spoken_summary_leaves_short_prose_unchanged():
    # Below the cap: returned verbatim, with no truncation ellipsis appended.
    assert spoken_summary("Done — added the flag.") == "Done — added the flag."


def test_build_voice_session_readback_tracks_tts_availability(monkeypatch):
    monkeypatch.setattr(voicemod.tts_session, "is_available", lambda: True)
    assert build_voice_session("k").readback is True
    monkeypatch.setattr(voicemod.tts_session, "is_available", lambda: False)
    assert build_voice_session("k").readback is False
