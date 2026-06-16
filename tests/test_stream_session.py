"""`assembly stream` session lifecycle + transcript-saving tests.

The --system-audio family lives in test_stream_system_audio.py; this file covers the
shared StreamSession plumbing (the "Listening…" latch, renderer teardown) and the
--save-transcript/--save-dir file writer.
"""

import types
from datetime import datetime


def _turn(text, *, speaker_label=None):
    return types.SimpleNamespace(transcript=text, end_of_turn=True, speaker_label=speaker_label)


def _save_plan(tmp_path, *, auto_name=False, write_note=False, name: str | None = "Meeting"):
    from aai_cli.streaming.savedir import SaveDirPlan

    return SaveDirPlan(
        save_dir=tmp_path / "rec",
        now=datetime(2026, 6, 16, 14, 30, 5),
        name=name,
        auto_name=auto_name,
        write_note=write_note,
    )


def test_save_dir_finalize_passes_recorded_metadata(monkeypatch, tmp_path):
    # A --save-dir run records each finalized turn's text + diarized speaker and the
    # wall-clock duration, then hands them to write_outputs once streaming ends. Pins
    # the speaker dedupe, the turn count, and the injected-clock duration.
    import io

    from aai_cli.streaming import savedir as savedir_mod
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        savedir_mod, "write_outputs", lambda plan, **kw: captured.update(kw) or plan.paths
    )
    monkeypatch.setattr(
        session_mod.client,
        "stream_audio",
        lambda api_key, source, *, on_turn, **k: [
            b"".join(source),
            on_turn(_turn("hello", speaker_label="A")),
            on_turn(_turn("again", speaker_label="A")),  # same speaker -> deduped
            on_turn(_turn("bye", speaker_label="B")),
        ],
    )
    ticks = iter([100.0, 107.0])
    session = session_mod.StreamSession(
        api_key="sk",
        base_flags={"speech_model": "u3-rt-pro"},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_audio=tmp_path / "out.wav",
        save_plan=_save_plan(tmp_path),
        clock=lambda: next(ticks),
    )
    session.run([b"\x00\x00"], 16000)

    assert captured["speakers"] == ["A", "B"]
    assert captured["turns"] == 3
    assert captured["duration_seconds"] == 7  # 107.0 - 100.0
    assert captured["title"] is None  # no --auto-name
    assert captured["note"] is None  # no --llm note
    assert captured["audio"] == [tmp_path / "out.wav"]  # the single teed WAV is handed on


def test_audio_files_lists_per_channel_or_single_or_none(tmp_path):
    # _audio_files reports what finalize should rename/record: the per-channel WAVs under
    # --system-audio, the lone save_audio otherwise, or nothing under --no-save-audio.
    import io

    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer

    def _session(**kw):
        return session_mod.StreamSession(
            api_key="sk",
            base_flags={},
            overrides=None,
            config_file=None,
            renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
            follow=None,
            llm_prompts=[],
            model="m",
            max_tokens=1,
            **kw,
        )

    you, system = tmp_path / "you.wav", tmp_path / "system.wav"
    assert _session(save_audio=tmp_path / "a.wav")._audio_files() == [tmp_path / "a.wav"]
    assert _session(save_audio_by_label={"you": you, "system": system})._audio_files() == [
        you,
        system,
    ]
    assert _session()._audio_files() == []


def test_save_dir_finalize_derives_title_and_note(monkeypatch, tmp_path):
    # --auto-name derives the title from the transcript via the LLM, and --llm's final
    # answer is handed to write_outputs as the note.
    import io

    from aai_cli.streaming import savedir as savedir_mod
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer
    from aai_cli.ui.follow import FollowRenderer

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        savedir_mod, "write_outputs", lambda plan, **kw: captured.update(kw) or plan.paths
    )
    monkeypatch.setattr(savedir_mod, "derive_title", lambda *a, **k: "Derived Title")
    monkeypatch.setattr(session_mod.llm, "run_chain", lambda *a, **k: "the summary")
    monkeypatch.setattr(
        session_mod.client,
        "stream_audio",
        lambda api_key, source, *, on_turn, **k: [b"".join(source), on_turn(_turn("hi"))],
    )
    session = session_mod.StreamSession(
        api_key="sk",
        base_flags={"speech_model": "u3-rt-pro"},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=FollowRenderer(json_mode=True),
        llm_prompts=["summarize"],
        model="m",
        max_tokens=1,
        save_plan=_save_plan(tmp_path, auto_name=True, write_note=True, name=None),
        llm_interval=0.0,
    )
    session.run([b"\x00\x00"], 16000)

    assert captured["title"] == "Derived Title"
    assert captured["note"] == "the summary"


def test_save_dir_skips_title_when_transcript_is_empty(monkeypatch, tmp_path):
    # --auto-name with zero finalized turns has nothing to title, so derive_title is
    # skipped and the file keeps its timestamp stem (pins the `auto_name and text` guard).
    import io

    from aai_cli.streaming import savedir as savedir_mod
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        savedir_mod, "write_outputs", lambda plan, **kw: captured.update(kw) or plan.paths
    )
    monkeypatch.setattr(savedir_mod, "derive_title", lambda *a, **k: "Should Not Be Used")
    monkeypatch.setattr(
        session_mod.client,
        "stream_audio",
        lambda api_key, source, *, on_turn, **k: b"".join(source),  # no turns fired
    )
    session = session_mod.StreamSession(
        api_key="sk",
        base_flags={"speech_model": "u3-rt-pro"},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_plan=_save_plan(tmp_path, auto_name=True, name=None),
    )
    session.run([b"\x00\x00"], 16000)

    assert captured["title"] is None  # no transcript -> no LLM title call
    assert captured["turns"] == 0


def test_finalize_uses_zero_duration_when_capture_never_started(monkeypatch, tmp_path):
    # If the capture window never opened (stream_one not reached), the sidecar duration is
    # 0, not a bogus value (pins the `0 if _capture_start is None` literal).
    import io

    from aai_cli.streaming import savedir as savedir_mod
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        savedir_mod, "write_outputs", lambda plan, **kw: captured.update(kw) or plan.paths
    )
    plan = _save_plan(tmp_path)
    session = session_mod.StreamSession(
        api_key="sk",
        base_flags={},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_plan=plan,
    )
    session._finalize_save_dir(plan)  # no run() -> _capture_start stayed None

    assert captured["duration_seconds"] == 0


def test_save_dir_auto_name_failure_keeps_recording(monkeypatch, tmp_path):
    # A failed --auto-name title call must not lose the (already-saved) recording: the
    # error is warned and write_outputs still runs, with no title.
    import io

    from aai_cli.core.errors import APIError
    from aai_cli.streaming import savedir as savedir_mod
    from aai_cli.streaming import session as session_mod
    from aai_cli.streaming.render import StreamRenderer

    def boom(*_a, **_k):
        raise APIError("gateway down")

    captured: dict[str, object] = {}
    monkeypatch.setattr(savedir_mod, "derive_title", boom)
    monkeypatch.setattr(
        savedir_mod, "write_outputs", lambda plan, **kw: captured.update(kw) or plan.paths
    )
    monkeypatch.setattr(
        session_mod.client,
        "stream_audio",
        lambda api_key, source, *, on_turn, **k: [b"".join(source), on_turn(_turn("hi"))],
    )
    session = session_mod.StreamSession(
        api_key="sk",
        base_flags={"speech_model": "u3-rt-pro"},
        overrides=None,
        config_file=None,
        renderer=StreamRenderer(json_mode=True, out=io.StringIO()),
        follow=None,
        llm_prompts=[],
        model="m",
        max_tokens=1,
        save_plan=_save_plan(tmp_path, auto_name=True, name=None),
    )
    session.run([b"\x00\x00"], 16000)

    assert captured["title"] is None  # finalize still ran, just without a derived title


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
