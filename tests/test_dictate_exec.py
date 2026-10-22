"""Direct tests of the `assembly dictate` options/run seam (dictate_exec).

The session is driven by constructing DictateOptions and injecting the three
boundaries — stop_on_terminate (a scripted SIGTERM latch), MicrophoneSource
(canned PCM), and dictation.transcribe_live (recorded calls) — so no test needs a
real signal, microphone, or network.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json

import pytest
import typer

from aai_cli.app.context import AppState
from aai_cli.commands.dictate import _exec as dictate_exec
from aai_cli.core import choices, config, dictation
from aai_cli.core.errors import NotAuthenticated, UsageError

DICTATE_DEFAULTS = dictate_exec.DictateOptions(
    language=None,
    prompt=None,
    keyterms=None,
    device=None,
    once=False,
    max_seconds=120.0,
)

# One ~100 ms chunk of 16 kHz PCM16.
CHUNK = b"\x01\x00" * 1600

# The verbatim transcript differs from the cleanup rewrite, so a test can tell which
# one was printed: the human default is the rewrite ("hello world").
RESULT = dictation.DictationResult(
    text="um, hello world",
    llm_response="hello world",
    llm_error=None,
    confidence=0.9,
    audio_duration_ms=1500,
    session_id="sess-1",
)


class FakeStop:
    """A scripted stop_on_terminate: the yielded predicate pops the next scripted
    bool per poll (an exhausted script reads as False = keep recording), so a True
    stands in for a SIGTERM arriving at that point in the capture."""

    def __init__(self, script):
        self.script = list(script)
        self.polls = 0
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self._poll

    def __exit__(self, *exc):
        self.exited = True

    def _poll(self):
        self.polls += 1
        return self.script.pop(0) if self.script else False


class RaisingStop(FakeStop):
    """A stop latch whose poll raises KeyboardInterrupt — i.e. SIGINT (Ctrl-C)
    arriving mid-recording."""

    def _poll(self):
        raise KeyboardInterrupt


@pytest.fixture
def seams(monkeypatch):
    """Wire all three boundaries; returns the mutable harness state."""
    config.set_api_key("default", "sk_live")
    harness = {"stop": FakeStop([]), "chunks": [CHUNK, CHUNK], "mic": {}, "calls": []}

    monkeypatch.setattr(dictate_exec, "stop_on_terminate", lambda: harness["stop"])

    def fake_mic(*, target_rate, device=None, on_open=None):
        harness["mic"].update(target_rate=target_rate, device=device)
        if on_open is not None:
            on_open()
        return iter(harness["chunks"])

    monkeypatch.setattr(dictate_exec, "MicrophoneSource", fake_mic)

    harness["result"] = RESULT

    def fake_transcribe(api_key, pcm, *, sample_rate, channels=1, config=None):
        harness["calls"].append(
            {
                "api_key": api_key,
                "pcm": pcm,
                "sample_rate": sample_rate,
                "channels": channels,
                "config": config,
            }
        )
        return harness["result"]

    monkeypatch.setattr(dictate_exec.dictation, "transcribe_live", fake_transcribe)
    return harness


def _run(opts=DICTATE_DEFAULTS, state=None, *, json_mode=False):
    dictate_exec.run_dictate(opts, state or AppState(), json_mode=json_mode)


def test_options_are_immutable():
    field_name = dataclasses.fields(DICTATE_DEFAULTS)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DICTATE_DEFAULTS, field_name, None)


def test_records_then_prints_bare_transcript(seams, capsys):
    # Recording auto-starts; the first poll sees no SIGTERM, a SIGTERM after the
    # second chunk stops the capture, then dictate exits.
    seams["stop"] = FakeStop([False, True])
    _run()
    # Both chunks were captured and uploaded as one utterance at the resampled rate.
    assert seams["calls"] == [
        {
            "api_key": "sk_live",
            "pcm": CHUNK + CHUNK,
            "sample_rate": 16000,
            "channels": 1,
            "config": dictation.DictationConfig(),
        }
    ]
    captured = capsys.readouterr()
    # Human mode: the bare text on stdout (pipe-friendly), not a JSON object.
    assert captured.out.strip() == "hello world"
    # The mic-open note fires on stderr and names the SIGTERM stop.
    assert "send SIGTERM to transcribe" in captured.err
    assert seams["mic"] == {"target_rate": 16000, "device": None}
    assert seams["stop"].entered and seams["stop"].exited  # handler installed + restored
    # Polled once per captured chunk: False after chunk 1, True after chunk 2.
    assert seams["stop"].polls == 2


def test_json_mode_emits_one_ndjson_object_per_utterance(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "type": "utterance",
        "text": "um, hello world",
        "llm_response": "hello world",
        "llm_error": None,
        "confidence": 0.9,
        "audio_duration_ms": 1500,
        "session_id": "sess-1",
    }
    # --json keeps stderr machine-readable: no human hints.
    assert captured.err == ""


def test_output_json_folds_into_ndjson_without_the_json_flag(seams, capsys):
    # -o json must enable NDJSON on its own (json_mode stays the --json flag,
    # which is False here) — proving the -o/--output resolution runs.
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.json))
    assert json.loads(capsys.readouterr().out)["text"] == "um, hello world"


def test_output_text_emits_bare_transcript(seams, capsys):
    # -o text is the explicit spelling of the human default: bare text, no JSON.
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.text))
    out = capsys.readouterr().out
    assert out.strip() == "hello world"
    assert "{" not in out


def test_output_text_conflicts_with_json_flag(seams):
    # --json + -o text are contradictory output shapes: a clean usage error,
    # the same as `stream`/`agent`.
    seams["stop"] = FakeStop([True])
    with pytest.raises(UsageError):
        _run(
            dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.text),
            json_mode=True,
        )


def test_quiet_suppresses_the_recording_hint(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(state=AppState(quiet=True))
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
    assert captured.err == ""


def test_records_one_utterance_then_exits(seams, capsys):
    # A SIGTERM on the first poll stops the capture and dictate exits after one
    # utterance — it never starts a second recording, and the rest of the stop
    # script is left undrained.
    seams["stop"] = FakeStop([True, True, True])
    _run()
    assert len(seams["calls"]) == 1
    assert seams["stop"].script  # ended on the single utterance, not by draining the script
    assert seams["stop"].polls == 1
    assert capsys.readouterr().out.strip() == "hello world"


def test_once_flag_is_a_deprecated_noop_that_warns(seams, capsys):
    # --once is kept only so old scripts don't break: it does nothing (single
    # utterance is the default) but warns that it can be dropped.
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, once=True))
    assert len(seams["calls"]) == 1
    assert "--once is now the default" in capsys.readouterr().err


def test_once_warning_is_silenced_by_quiet(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, once=True), state=AppState(quiet=True))
    assert "--once" not in capsys.readouterr().err


def test_sigterm_stops_recording_and_transcribes(seams):
    # A SIGTERM after the first chunk stops the auto-started recording and the
    # captured utterance is still transcribed.
    seams["stop"] = FakeStop([True])
    _run()
    assert len(seams["calls"]) == 1
    assert seams["calls"][0]["pcm"] == CHUNK  # stopped after the first chunk


def test_no_signal_does_not_stop_capture(seams):
    # A poll that reports no SIGTERM keeps recording; only a True (or the cap)
    # ends the capture.
    seams["stop"] = FakeStop([False, True])
    seams["chunks"] = [CHUNK, CHUNK, CHUNK]
    _run()
    assert seams["calls"][0]["pcm"] == CHUNK + CHUNK


def test_recording_stops_at_the_duration_cap(seams):
    # 0.2 s at 16 kHz PCM16 = 6400 bytes = exactly two chunks; no SIGTERM ever
    # arrives, so only the cap can stop the capture.
    seams["stop"] = FakeStop([])
    seams["chunks"] = [CHUNK] * 5
    _run(dataclasses.replace(DICTATE_DEFAULTS, max_seconds=0.2))
    assert len(seams["calls"]) == 1
    assert seams["calls"][0]["pcm"] == CHUNK + CHUNK


def test_recording_closes_the_mic_generator(seams):
    closed = []

    def chunk_gen():
        try:
            yield CHUNK
            yield CHUNK
            yield CHUNK
        finally:
            closed.append(True)

    seams["stop"] = FakeStop([True])
    seams["chunks"] = chunk_gen()
    _run()
    assert closed == [True]  # the device-releasing cleanup ran at stop, not at GC


def test_empty_recording_is_skipped_with_a_warning(seams, capsys):
    # An instant SIGTERM captures nothing; the server refuses an empty audio part.
    seams["stop"] = FakeStop([True])
    seams["chunks"] = [b""]
    _run()
    assert seams["calls"] == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Nothing was recorded" in captured.err


def test_a_very_short_recording_is_still_transcribed(seams):
    # The live endpoint has no minimum duration, so even one sample goes up.
    seams["stop"] = FakeStop([True])
    seams["chunks"] = [b"\x01\x00"]
    _run()
    assert len(seams["calls"]) == 1


def test_flags_map_onto_the_dictation_config(seams):
    seams["stop"] = FakeStop([True])
    _run(
        dataclasses.replace(
            DICTATE_DEFAULTS,
            language="en, es",
            prompt="A doctor dictating a visit note.",
            keyterms=["AssemblyAI"],
            instruction="Format as a bulleted list",
            rewrite_mode=choices.RewriteMode.smart,
        )
    )
    assert seams["calls"][0]["config"] == dictation.DictationConfig(
        language_codes=["en", "es"],
        stt_prompt="A doctor dictating a visit note.",
        keyterms=["AssemblyAI"],
        llm_instruction="Format as a bulleted list",
        rewrite_mode="smart",
    )


def test_blank_language_reads_as_unset(seams):
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, language=" , "))
    assert seams["calls"][0]["config"].language_codes is None


def test_deprecated_word_boost_joins_the_keyterms_and_warns(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, keyterms=["A"], word_boost=["B"]))
    assert seams["calls"][0]["config"].keyterms == ["A", "B"]
    assert "--word-boost is deprecated; use --keyterm." in capsys.readouterr().err


def test_word_boost_warning_is_silenced_by_quiet(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, word_boost=["B"]), AppState(quiet=True))
    assert "deprecated" not in capsys.readouterr().err


def test_verbatim_prints_the_unedited_transcript(seams, capsys):
    seams["stop"] = FakeStop([True])
    _run(dataclasses.replace(DICTATE_DEFAULTS, verbatim=True))
    assert capsys.readouterr().out.strip() == "um, hello world"


def test_failed_rewrite_falls_back_to_the_transcript_with_a_warning(seams, capsys):
    seams["stop"] = FakeStop([True])
    seams["result"] = dataclasses.replace(RESULT, llm_response=None, llm_error="timeout")
    _run()
    captured = capsys.readouterr()
    assert captured.out.strip() == "um, hello world"
    assert "The cleanup rewrite failed (timeout)" in captured.err


def test_verbatim_skips_the_rewrite_failure_warning(seams, capsys):
    seams["stop"] = FakeStop([True])
    seams["result"] = dataclasses.replace(RESULT, llm_response=None, llm_error="error")
    _run(dataclasses.replace(DICTATE_DEFAULTS, verbatim=True))
    assert "rewrite failed" not in capsys.readouterr().err


def test_transcription_runs_under_the_status_spinner(seams, monkeypatch):
    seen = {}

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet=False):
        seen.update(message=message, json_mode=json_mode, quiet=quiet)
        yield

    monkeypatch.setattr(dictate_exec.output, "status", fake_status)
    seams["stop"] = FakeStop([True])
    _run(state=AppState(quiet=True))
    assert seen == {"message": "Transcribing…", "json_mode": False, "quiet": True}


def test_ctrl_c_exits_with_cancel_code(seams):
    stop = RaisingStop([])
    seams["stop"] = stop
    # Ctrl-C / SIGINT cancels dictation: exit 130 (distinct from SIGTERM, which
    # finishes with 0).
    with pytest.raises(typer.Exit) as exc:
        _run()
    assert exc.value.exit_code == 130
    assert stop.exited  # the with-block unwound, restoring the previous handler


def test_credentials_are_resolved_before_recording(seams, monkeypatch):
    # No key is configured: the missing-credentials error must surface before the
    # mic is ever opened (don't capture audio we can't transcribe).
    config.clear_api_key("default")
    opened = []
    monkeypatch.setattr(
        dictate_exec, "MicrophoneSource", lambda **_kw: opened.append(True) or iter([])
    )
    with pytest.raises(NotAuthenticated):
        _run()
    assert opened == []  # the mic was never opened
