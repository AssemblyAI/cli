"""Ordering of overlapping ``stream --llm`` refreshes.

The chain runs outside the session lock so a second source's turns keep flowing during a
refresh, which means two refreshes can be in flight at once and finish out of order.
These tests script that interleaving with events (no sleeps, sockets, or microphone):
the first ``run_chain`` call parks until released, so a later, longer refresh lands first.
"""

import io
import threading
import types
from datetime import datetime

import pytest

from aai_cli.core.errors import APIError
from aai_cli.streaming import naming, savedir
from aai_cli.streaming.render import StreamRenderer
from aai_cli.streaming.session import StreamSession
from aai_cli.ui.follow import FollowRenderer

_WAIT = 5.0


class _GatedChain:
    """A fake ``llm.run_chain`` whose first call blocks until ``release`` is set."""

    def __init__(self, *, fail_first=False):
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_first = fail_first

    def __call__(self, api_key, prompts, *, transcript_text, model, max_tokens):
        index = len(self.calls)
        self.calls.append(transcript_text)
        if index == 0:
            self.entered.set()
            assert self.release.wait(_WAIT)
            if self.fail_first:
                raise APIError("gateway down")
        return f"{transcript_text}#{index + 1}"


def _turn(text):
    return types.SimpleNamespace(transcript=text, end_of_turn=True, speaker_label=None)


def _session(monkeypatch, emitted, chain, *, interval=0.0, clock=lambda: 0.0, **extra):
    monkeypatch.setattr("aai_cli.ui.follow.output.emit_ndjson", emitted.append)
    monkeypatch.setattr("aai_cli.streaming.session.llm.run_chain", chain)
    return StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=FollowRenderer(json_mode=True),
        llm_prompts=["sum"],
        model="m",
        max_tokens=10,
        llm_interval=interval,
        clock=clock,
        **extra,
    )


def _start(target):
    errors: list[Exception] = []

    def run():
        try:
            target()
        except (AssertionError, APIError) as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def _finish(thread, errors):
    thread.join(_WAIT)
    assert not thread.is_alive()
    assert errors == []


def test_late_older_refresh_does_not_replace_newer_answer(monkeypatch):
    emitted: list[dict] = []
    chain = _GatedChain()
    session = _session(monkeypatch, emitted, chain)
    slow = _start(lambda: session.on_turn(_turn("a")))  # refresh over turn 1 parks
    assert chain.entered.wait(_WAIT)
    session.on_turn(_turn("b"))  # refresh over turns 1-2 lands first
    chain.release.set()
    _finish(*slow)
    assert emitted == [{"type": "answer", "turns": 2, "output": "a b#2"}]
    assert session._last_answer == "a b#2"
    session._maybe_summarize(final=True)  # both turns already published -> no re-run
    assert chain.calls == ["a", "a b"]
    assert len(emitted) == 1


def test_refresh_outlasting_the_interval_cannot_regress_the_answer(monkeypatch):
    now = {"t": 1000.0}
    emitted: list[dict] = []
    chain = _GatedChain()
    session = _session(monkeypatch, emitted, chain, interval=30.0, clock=lambda: now["t"])
    slow = _start(lambda: session.on_turn(_turn("a")))
    assert chain.entered.wait(_WAIT)
    now["t"] = 1040.0  # the slow request has outlived the 30s window
    session.on_turn(_turn("b"))
    chain.release.set()
    _finish(*slow)
    now["t"] = 1050.0
    session.on_turn(_turn("c"))  # inside the new window -> throttled
    session._maybe_summarize(final=True)  # closing flush covers the tail
    assert [(e["turns"], e["output"]) for e in emitted] == [(2, "a b#2"), (3, "a b c#3")]
    assert session._last_answer == "a b c#3"


def test_closing_flush_does_not_wait_on_an_unfinished_refresh(monkeypatch):
    # Ctrl-C can end the stream while a reader thread is still inside run_chain; the
    # closing flush must produce an answer for the full transcript itself.
    emitted: list[dict] = []
    chain = _GatedChain()
    session = _session(monkeypatch, emitted, chain)
    slow = _start(lambda: session.on_turn(_turn("a")))
    assert chain.entered.wait(_WAIT)
    session._maybe_summarize()  # mid-stream, the in-flight refresh already claims turn 1
    assert chain.calls == ["a"]
    session._maybe_summarize(final=True)
    assert session._last_answer == "a#2"
    chain.release.set()
    _finish(*slow)
    # The parked refresh covered the same turns, so it lands as a duplicate and is dropped.
    assert emitted == [{"type": "answer", "turns": 1, "output": "a#2"}]
    assert session._last_answer == "a#2"


def test_failed_older_refresh_keeps_newer_answer_and_still_fails_close(monkeypatch, capsys):
    emitted: list[dict] = []
    chain = _GatedChain(fail_first=True)
    session = _session(monkeypatch, emitted, chain)
    slow = _start(lambda: session.on_turn(_turn("a")))
    assert chain.entered.wait(_WAIT)
    session.on_turn(_turn("b"))
    chain.release.set()
    _finish(*slow)
    assert emitted == [{"type": "answer", "turns": 2, "output": "a b#2"}]
    assert session._last_answer == "a b#2"
    assert "gateway down" in capsys.readouterr().err
    session.on_turn(_turn("c"))  # the latched error stops further refreshes
    assert chain.calls == ["a", "a b"]
    with pytest.raises(APIError):
        session._maybe_summarize(final=True)


def test_parallel_sources_save_note_for_latest_transcript(monkeypatch, tmp_path):
    # Drives the real run_parallel -> on_turn -> closing flush -> --save-dir finalize path,
    # faking only the two network boundaries (the streaming socket and the LLM chain).
    emitted: list[dict] = []
    chain = _GatedChain()
    plan = savedir.SaveDirPlan(
        save_dir=tmp_path,
        now=datetime(2026, 1, 2, 3, 4, 5),
        name="sync",
        auto_name=False,
        write_note=True,
    )
    naming.ensure_dir(plan.paths.directory)

    def fake_stream_audio(api_key, audio, *, params, on_begin, on_turn, on_termination):
        source = next(iter(audio))
        if source == b"system":
            on_turn(_turn("first"))  # parks inside the chain
        else:
            assert chain.entered.wait(_WAIT)
            on_turn(_turn("second"))
            chain.release.set()

    monkeypatch.setattr("aai_cli.streaming.session.client.stream_audio", fake_stream_audio)
    session = _session(
        monkeypatch, emitted, chain, save_plan=plan, save_transcript=plan.paths.transcript
    )
    session.run_parallel([("system", [b"system"], 16000), ("you", [b"you"], 16000)])
    latest = " ".join(session.transcript)
    assert [e["turns"] for e in emitted] == [2]
    assert session._last_answer == f"{latest}#2"
    assert plan.paths.note.read_text(encoding="utf-8") == f"{latest}#2\n"
    assert plan.paths.transcript.read_text(encoding="utf-8").splitlines() == session.transcript
