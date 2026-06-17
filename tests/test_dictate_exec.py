"""Direct tests of the `assembly dictate` options/run seam (dictate_exec).

The session is driven by constructing DictateOptions and injecting the three
boundaries — TerminalKeys (scripted keys), MicrophoneSource (canned PCM), and
sync_stt.transcribe_pcm (recorded calls) — so no test needs a real terminal,
microphone, or network.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json

import pytest
import typer

from aai_cli.app.context import AppState
from aai_cli.commands.dictate import _exec as dictate_exec
from aai_cli.core import choices, config, sync_stt
from aai_cli.core.errors import CLIError, UsageError

DICTATE_DEFAULTS = dictate_exec.DictateOptions(
    language=None,
    prompt=None,
    word_boost=None,
    device=None,
    once=False,
    max_seconds=120.0,
)

# One ~100 ms chunk of 16 kHz PCM16 — comfortably above the 80 ms upload floor.
CHUNK = b"\x01\x00" * 1600

RESULT = sync_stt.SyncTranscript(
    text="hello world", confidence=0.9, audio_duration_ms=1500, session_id="sess-1"
)


class FakeKeys:
    """A scripted TerminalKeys: each read() pops the next key (None = no key yet);
    an exhausted script reads as EOF, which ends the session."""

    def __init__(self, script):
        self.script = list(script)
        self.timeouts = []
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc):
        self.exited = True

    def read(self, timeout):
        self.timeouts.append(timeout)
        return self.script.pop(0) if self.script else None


class RaisingKeys(FakeKeys):
    def read(self, timeout):
        raise KeyboardInterrupt


@pytest.fixture
def seams(monkeypatch):
    """Wire all three boundaries; returns the mutable harness state."""
    config.set_api_key("default", "sk_live")
    harness = {"keys": FakeKeys([]), "chunks": [CHUNK, CHUNK], "mic": {}, "calls": []}

    monkeypatch.setattr(dictate_exec, "TerminalKeys", lambda: harness["keys"])

    def fake_mic(*, target_rate, device=None, on_open=None):
        harness["mic"].update(target_rate=target_rate, device=device)
        if on_open is not None:
            on_open()
        return iter(harness["chunks"])

    monkeypatch.setattr(dictate_exec, "MicrophoneSource", fake_mic)

    def fake_transcribe(api_key, pcm, *, sample_rate, channels=1, **kwargs):
        harness["calls"].append(
            {"api_key": api_key, "pcm": pcm, "sample_rate": sample_rate, "channels": channels}
            | kwargs
        )
        return RESULT

    monkeypatch.setattr(dictate_exec.sync_stt, "transcribe_pcm", fake_transcribe)
    return harness


def _run(opts=DICTATE_DEFAULTS, state=None, *, json_mode=False):
    dictate_exec.run_dictate(opts, state or AppState(), json_mode=json_mode)


def test_options_are_immutable():
    field_name = dataclasses.fields(DICTATE_DEFAULTS)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(DICTATE_DEFAULTS, field_name, None)


def test_records_then_prints_bare_transcript(seams, capsys):
    # Recording auto-starts; the in-recording poll sees nothing after chunk 1,
    # Enter after chunk 2 stops the capture, then dictate exits.
    seams["keys"] = FakeKeys([None, "\r"])
    _run()
    # Both chunks were captured and uploaded as one utterance at the resampled rate.
    assert seams["calls"] == [
        {
            "api_key": "sk_live",
            "pcm": CHUNK + CHUNK,
            "sample_rate": 16000,
            "channels": 1,
            "language_code": None,
            "prompt": None,
            "word_boost": None,
        }
    ]
    captured = capsys.readouterr()
    # Human mode: the bare text on stdout (pipe-friendly), not a JSON object.
    assert captured.out.strip() == "hello world"
    # The mic-open note fires on stderr; there is no interactive start prompt.
    assert "Recording — press Enter to stop" in captured.err
    assert "start recording" not in captured.err
    assert seams["mic"] == {"target_rate": 16000, "device": None}
    assert seams["keys"].entered and seams["keys"].exited  # terminal restored
    # Recording auto-started, so every read is the zero-timeout in-recording poll
    # — no blocking idle read(None) waiting for a start keypress.
    assert seams["keys"].timeouts == [0, 0]


def test_json_mode_emits_one_ndjson_object_per_utterance(seams, capsys):
    seams["keys"] = FakeKeys(["\r"])
    _run(json_mode=True)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "type": "utterance",
        "text": "hello world",
        "confidence": 0.9,
        "audio_duration_ms": 1500,
        "session_id": "sess-1",
    }
    # --json keeps stderr machine-readable: no human hints.
    assert captured.err == ""


def test_output_json_folds_into_ndjson_without_the_json_flag(seams, capsys):
    # -o json must enable NDJSON on its own (json_mode stays the --json flag,
    # which is False here) — proving the -o/--output resolution runs.
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.json))
    assert json.loads(capsys.readouterr().out)["text"] == "hello world"


def test_output_text_emits_bare_transcript(seams, capsys):
    # -o text is the explicit spelling of the human default: bare text, no JSON.
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.text))
    out = capsys.readouterr().out
    assert out.strip() == "hello world"
    assert "{" not in out


def test_output_text_conflicts_with_json_flag(seams):
    # --json + -o text are contradictory output shapes: a clean usage error,
    # the same as `stream`/`agent`.
    seams["keys"] = FakeKeys(["\r", "\r"])
    with pytest.raises(UsageError):
        _run(
            dataclasses.replace(DICTATE_DEFAULTS, output_field=choices.TextOrJson.text),
            json_mode=True,
        )


def test_quiet_suppresses_the_interactive_hints(seams, capsys):
    seams["keys"] = FakeKeys(["\r"])
    _run(state=AppState(quiet=True))
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
    assert captured.err == ""


def test_auto_starts_recording_then_exits_after_one_utterance(seams, capsys):
    # Recording auto-starts: a single read(0) pops the Enter that stops the
    # capture, then dictate exits — no blocking idle read(None) waiting for a
    # start keypress, and the rest of the key script is left undrained.
    seams["keys"] = FakeKeys(["\r", "\r", "\r"])
    _run()
    assert len(seams["calls"]) == 1
    assert seams["keys"].script  # ended on the single utterance, not by draining keys
    assert seams["keys"].timeouts == [0]
    captured = capsys.readouterr()
    assert captured.out.strip() == "hello world"
    # The mic-open note fires immediately; there is no interactive start prompt.
    assert "Recording — press Enter to stop" in captured.err
    assert "start recording" not in captured.err


def test_once_flag_is_a_deprecated_noop_that_warns(seams, capsys):
    # --once is kept only so old scripts don't break: it does nothing (single
    # utterance is the default) but warns that it can be dropped.
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, once=True))
    assert len(seams["calls"]) == 1
    assert "--once is now the default" in capsys.readouterr().err


def test_once_warning_is_silenced_by_quiet(seams, capsys):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, once=True), state=AppState(quiet=True))
    assert "--once" not in capsys.readouterr().err


@pytest.mark.parametrize("quit_key", ["q", "Q", "\x1b", "\x04"])
def test_quit_keys_stop_recording_and_transcribe(seams, quit_key):
    # No idle prompt anymore: q / Esc / Ctrl-D stop the auto-started recording
    # and the captured utterance is still transcribed.
    seams["keys"] = FakeKeys([quit_key])
    _run()
    assert len(seams["calls"]) == 1
    assert seams["calls"][0]["pcm"] == CHUNK  # stopped after the first chunk


def test_space_also_stops_recording(seams):
    seams["keys"] = FakeKeys([" "])
    _run()
    assert len(seams["calls"]) == 1


def test_unbound_keys_during_recording_do_not_stop_capture(seams):
    # A stray keystroke mid-utterance is ignored; only Enter/Space (or a quit
    # key) ends the capture.
    seams["keys"] = FakeKeys(["x", "\r"])
    seams["chunks"] = [CHUNK, CHUNK, CHUNK]
    _run()
    assert seams["calls"][0]["pcm"] == CHUNK + CHUNK


def test_recording_stops_at_the_duration_cap(seams):
    # 0.2 s at 16 kHz PCM16 = 6400 bytes = exactly two chunks; the poll never
    # reports a key, so only the cap can stop the capture.
    seams["keys"] = FakeKeys([])
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

    seams["keys"] = FakeKeys(["\r"])
    seams["chunks"] = chunk_gen()
    _run()
    assert closed == [True]  # the device-releasing cleanup ran at stop, not at GC


@pytest.mark.parametrize("size", [200, 2558])  # 2558: just under the exact 2560-byte floor
def test_too_short_recording_is_skipped_with_a_warning(seams, capsys, size):
    seams["keys"] = FakeKeys(["\r"])
    seams["chunks"] = [b"\x01" * size]  # below 80 ms of 16 kHz PCM16 (2560 bytes)
    _run()
    assert seams["calls"] == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "shorter than 80 ms" in captured.err


def test_recording_at_the_80ms_floor_is_transcribed(seams):
    seams["keys"] = FakeKeys(["\r"])
    seams["chunks"] = [b"\x01" * 2560]  # exactly 80 ms: allowed, not skipped
    _run()
    assert len(seams["calls"]) == 1


def test_language_and_boost_flags_are_forwarded(seams):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, language="es", word_boost=["AssemblyAI"]))
    assert seams["calls"][0]["language_code"] == "es"
    assert seams["calls"][0]["word_boost"] == ["AssemblyAI"]


def test_comma_separated_languages_become_a_list(seams):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, language="en, es"))
    assert seams["calls"][0]["language_code"] == ["en", "es"]


def test_blank_language_reads_as_unset(seams):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, language=" , "))
    assert seams["calls"][0]["language_code"] is None


def test_prompt_with_language_warns_that_language_is_ignored(seams, capsys):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, prompt="Verbatim.", language="es"))
    assert "--language is ignored when --prompt is set" in capsys.readouterr().err


def test_prompt_alone_is_forwarded_without_warning(seams, capsys):
    seams["keys"] = FakeKeys(["\r"])
    _run(dataclasses.replace(DICTATE_DEFAULTS, prompt="Verbatim."))
    assert seams["calls"][0]["prompt"] == "Verbatim."
    assert "ignored" not in capsys.readouterr().err


def test_transcription_runs_under_the_status_spinner(seams, monkeypatch):
    seen = {}

    @contextlib.contextmanager
    def fake_status(message, *, json_mode, quiet=False):
        seen.update(message=message, json_mode=json_mode, quiet=quiet)
        yield

    monkeypatch.setattr(dictate_exec.output, "status", fake_status)
    seams["keys"] = FakeKeys(["\r"])
    _run(state=AppState(quiet=True))
    assert seen == {"message": "Transcribing…", "json_mode": False, "quiet": True}


def test_ctrl_c_exits_with_cancel_code(seams):
    keys = RaisingKeys([])
    seams["keys"] = keys
    # Ctrl-C cancels dictation: exit 130 (distinct from `q`, which finishes with 0).
    with pytest.raises(typer.Exit) as exc:
        _run()
    assert exc.value.exit_code == 130
    assert keys.exited  # the with-block unwound, restoring the terminal


def test_terminal_is_validated_before_credentials(seams, monkeypatch):
    # No key is configured and TerminalKeys rejects the terminal: the usage
    # error must win, not NotAuthenticated (validation before credentials).
    config.clear_api_key("default")

    class NoTty:
        def __enter__(self):
            raise CLIError("This command needs an interactive terminal.", exit_code=2)

        def __exit__(self, *exc):
            return None

    monkeypatch.setattr(dictate_exec, "TerminalKeys", NoTty)
    with pytest.raises(CLIError) as exc:
        _run()
    assert exc.value.exit_code == 2
    assert "interactive terminal" in exc.value.message
