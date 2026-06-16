"""`assembly stream` session lifecycle + transcript-saving tests.

The --system-audio family lives in test_stream_system_audio.py; this file covers the
shared StreamSession plumbing (the "Listening…" latch, renderer teardown) and the
--save-transcript/--save-dir file writer.
"""

import types


def test_stream_session_listening_notice_latches(monkeypatch):
    # _listening_once must announce "Listening…" exactly once even if the first-audio
    # callback fires repeatedly (pins the `self._listening_started = True` latch).
    import io

    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.streaming.session import StreamSession

    renderer = StreamRenderer(json_mode=False, out=io.StringIO())
    calls = {"n": 0}
    monkeypatch.setattr(renderer, "listening", lambda: calls.__setitem__("n", calls["n"] + 1))
    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=renderer,
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
    )
    session._listening_once()
    session._listening_once()
    assert calls["n"] == 1


def test_stream_session_closes_renderer_on_error(monkeypatch):
    # When streaming raises mid-run, the live region must still be torn down (pins the
    # `if self.follow is None: self.renderer.close()` in the finally block).
    import io

    import pytest

    from aai_cli.core.errors import CLIError
    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.streaming.session import StreamSession

    renderer = StreamRenderer(json_mode=False, out=io.StringIO())
    closed = {"n": 0}
    monkeypatch.setattr(renderer, "close", lambda: closed.__setitem__("n", closed["n"] + 1))

    def boom(*_args, **_kwargs):
        raise CLIError("stream blew up")

    monkeypatch.setattr("aai_cli.commands.stream._exec.client.stream_audio", boom)
    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=renderer,
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
    )
    with pytest.raises(CLIError):
        session.run([b"\x00"], 16000)
    assert closed["n"] >= 1


def _saving_session(out, *, follow=None, llm_prompts=None):
    """A StreamSession wired with a transcript writer for the save-to-file tests."""
    import io

    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.streaming.session import StreamSession
    from aai_cli.streaming.transcript import TranscriptWriter

    session = StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=follow,
        llm_prompts=llm_prompts or [],
        model="m",
        max_tokens=1,
        save_transcript=out,
    )
    session._transcript_writer = TranscriptWriter(out)
    return session


def test_record_turn_saves_to_file_in_llm_mode(monkeypatch, tmp_path):
    # In --llm (follow) mode on_turn routes through _record_turn, which must also append
    # the finalized turn to the open --save-transcript file (pins the _save_line call there).
    from aai_cli.streaming import session as session_mod

    monkeypatch.setattr(session_mod.llm, "run_chain", lambda *a, **k: "answer")
    out = tmp_path / "notes.txt"
    summaries: list[str] = []
    session = _saving_session(
        out, follow=lambda answer, turns: summaries.append(answer), llm_prompts=["go"]
    )

    session.on_turn(types.SimpleNamespace(transcript="hello", end_of_turn=True, speaker_label=None))
    writer = session._transcript_writer
    assert writer is not None
    writer.close()

    assert out.read_text(encoding="utf-8") == "hello\n"
    assert session.transcript == ["hello"]  # still recorded for the --llm chain
    assert summaries == ["answer"]  # the chain refreshed off the saved turn


def test_guarded_closes_transcript_writer(monkeypatch, tmp_path):
    # The writer opened for a run is closed in _guarded's finally, even on a clean run
    # (pins the close() in the finally block — flush-per-turn alone wouldn't release it).
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.streaming.session import StreamSession

    closed = {"n": 0}
    real = session_mod.TranscriptWriter

    class SpyWriter(real):
        def close(self):
            closed["n"] += 1
            super().close()

    monkeypatch.setattr(session_mod, "TranscriptWriter", SpyWriter)
    monkeypatch.setattr(session_mod.client, "stream_audio", lambda *a, **k: list(a[1]))

    import io

    session = StreamSession(
        api_key="sk",
        base_flags={"speech_model": "u3-rt-pro"},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_transcript=tmp_path / "notes.txt",
    )
    session.run([b"\x00\x00"], 16000)
    assert closed["n"] == 1
