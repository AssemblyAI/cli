import io
import json
import types
from typing import TextIO, cast

import pytest

from aai_cli import theme
from aai_cli.streaming.render import StreamRenderer


def _turn(transcript, end_of_turn):
    return types.SimpleNamespace(transcript=transcript, end_of_turn=end_of_turn)


def _human(width=80, color_system=None):
    """A human-mode renderer writing to a forced-terminal themed console buffer."""
    buf = io.StringIO()
    console = theme.make_console(
        file=buf, force_terminal=True, width=width, color_system=color_system
    )
    return StreamRenderer(json_mode=False, out=buf, console=console), buf


def test_default_console_is_themed():
    buf = io.StringIO()
    r = StreamRenderer(json_mode=False, out=buf)
    # _console_obj builds via theme.make_console, so aai.* names resolve.
    r._console_obj().get_style("aai.brand")


# --- human mode (Rich) -----------------------------------------------------
def test_human_turn_shows_and_finalizes_text():
    r, buf = _human()
    r.turn(_turn("hello", False))
    r.turn(_turn("hello world", True))
    r.close()
    assert "hello world" in buf.getvalue()


def test_human_turn_labels_parallel_sources():
    r, buf = _human(color_system="truecolor")
    r.turn(_turn("hello from me", True), source="you")
    r.turn(_turn("hello from audio", True), source="system")
    r.close()
    out = buf.getvalue()
    assert "You:" in out
    assert "hello from me" in out
    assert "System:" in out
    assert "hello from audio" in out
    assert "\x1b[" in out


def test_text_mode_labels_sources_and_statuses_to_stderr():
    out = io.StringIO()
    err = io.StringIO()
    r = StreamRenderer(json_mode=False, text_mode=True, out=out, err=err)
    r.listening()
    r.turn(_turn("hello from me", True), source="you")
    r.stopped()
    assert out.getvalue() == "You: hello from me\n"
    assert "Listening" in err.getvalue()
    assert "Stopped." in err.getvalue()


def test_human_begin_is_silent_until_mic_opens():
    # The session opening (Begin) no longer prints "Listening…"; that waits for
    # the mic to actually open and start recording (renderer.listening()).
    r, buf = _human()
    r.begin(types.SimpleNamespace(id="x"))
    assert buf.getvalue() == ""


def test_human_listening_prints_notice():
    r, buf = _human()
    r.listening()
    out = buf.getvalue()
    assert "Listening" in out
    assert "Ctrl-C" in out


def test_human_long_partial_clears_wrapped_rows():
    # A partial wider than the terminal wraps; the next redraw must clear ALL
    # wrapped rows (Rich emits cursor-up), not stack copies on screen.
    r, buf = _human(width=20)
    r.turn(_turn("x" * 100, False))
    r.turn(_turn("y" * 100, False))
    assert "\x1b[1A" in buf.getvalue()  # moved up over the wrapped rows to clear them


def test_human_stopped_announced():
    r, buf = _human()
    r.stopped()
    assert "Stopped." in buf.getvalue()


def test_termination_silent_in_human_mode():
    r, buf = _human()
    r.termination(types.SimpleNamespace(audio_duration_seconds=3.0))
    assert buf.getvalue() == ""  # termination only surfaces in JSON


# --- json mode (plain NDJSON, unchanged) -----------------------------------
def test_json_mode_emits_ndjson_events():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.begin(types.SimpleNamespace(id="sess_1"))
    r.turn(_turn("hi", True))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0] == {"type": "begin", "id": "sess_1"}
    assert lines[1] == {"type": "turn", "transcript": "hi", "end_of_turn": True}


def test_json_mode_emits_source_when_labeled():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.begin(types.SimpleNamespace(id="sess_1"), source="system")
    r.turn(_turn("hi", True), source="you")
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0] == {"type": "begin", "id": "sess_1", "source": "system"}
    assert lines[1] == {
        "type": "turn",
        "transcript": "hi",
        "end_of_turn": True,
        "source": "you",
    }


def test_termination_json_emits_duration():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.termination(types.SimpleNamespace(audio_duration_seconds=12.5))
    assert json.loads(out.getvalue()) == {"type": "termination", "audio_duration_seconds": 12.5}


def test_close_is_noop_in_json_mode():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.turn(_turn("hi", False))
    before = out.getvalue()
    r.close()
    assert out.getvalue() == before


def test_json_emit_propagates_broken_pipe():
    class BrokenOut:
        def write(self, _text):
            raise BrokenPipeError("downstream closed")

        def flush(self):
            pass

    r = StreamRenderer(json_mode=True, out=cast(TextIO, BrokenOut()))
    # BrokenPipe must propagate so the command can stop cleanly (`| head`).
    with pytest.raises(BrokenPipeError):
        r.turn(_turn("hi", True))


def test_json_emit_swallows_non_pipe_errors():
    class FlakyOut:
        def write(self, _text):
            raise OSError("transient write error")

        def flush(self):
            pass

    r = StreamRenderer(json_mode=True, out=cast(TextIO, FlakyOut()))
    r.turn(_turn("hi", True))  # non-pipe write errors are non-fatal


def test_human_listening_notice_is_muted():
    r, buf = _human(color_system="truecolor")
    r.listening()
    assert "\x1b[" in buf.getvalue()  # muted styling emits ANSI


def test_listening_is_silent_in_json_mode():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.listening()
    assert out.getvalue() == ""  # the "Listening…" line is human-only
