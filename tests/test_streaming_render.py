import io
import json
import types

import pytest
from rich.console import Console

from assemblyai_cli.streaming.render import StreamRenderer


def _turn(transcript, end_of_turn):
    return types.SimpleNamespace(transcript=transcript, end_of_turn=end_of_turn)


def _human(width=80):
    """A human-mode renderer writing to a forced-terminal Rich console buffer."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=width, color_system=None)
    return StreamRenderer(json_mode=False, out=buf, console=console), buf


# --- human mode (Rich) -----------------------------------------------------
def test_human_turn_shows_and_finalizes_text():
    r, buf = _human()
    r.turn(_turn("hello", False))
    r.turn(_turn("hello world", True))
    r.close()
    assert "hello world" in buf.getvalue()


def test_human_begin_prints_notice():
    r, buf = _human()
    r.begin(types.SimpleNamespace(id="x"))
    assert "Ctrl-C" in buf.getvalue()


def test_human_long_partial_clears_wrapped_rows():
    # A partial wider than the terminal wraps; the next redraw must clear ALL
    # wrapped rows (Rich emits cursor-up), not stack copies on screen.
    r, buf = _human(width=20)
    r.turn(_turn("x" * 100, False))
    r.turn(_turn("y" * 100, False))
    assert "\x1b[1A" in buf.getvalue()  # moved up over the wrapped rows to clear them


def test_human_llm_line_rendered():
    r, buf = _human()
    r.turn(_turn("hola", True))
    r.llm("the summary")
    assert "the summary" in buf.getvalue()


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


def test_termination_json_emits_duration():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.termination(types.SimpleNamespace(audio_duration_seconds=12.5))
    assert json.loads(out.getvalue()) == {"type": "termination", "audio_duration_seconds": 12.5}


def test_llm_json_emits_event():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.llm("the summary")
    assert json.loads(out.getvalue()) == {"type": "llm", "content": "the summary"}


def test_llm_ignores_empty_content():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.llm("")
    assert out.getvalue() == ""


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

    r = StreamRenderer(json_mode=True, out=BrokenOut())
    # BrokenPipe must propagate so the command can stop cleanly (`| head`).
    with pytest.raises(BrokenPipeError):
        r.turn(_turn("hi", True))


def test_json_emit_swallows_non_pipe_errors():
    class FlakyOut:
        def write(self, _text):
            raise OSError("transient write error")

        def flush(self):
            pass

    r = StreamRenderer(json_mode=True, out=FlakyOut())
    r.turn(_turn("hi", True))  # non-pipe write errors are non-fatal
