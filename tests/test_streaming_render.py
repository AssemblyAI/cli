import io
import json
import types

from assemblyai_cli.streaming.render import StreamRenderer


def _turn(transcript, end_of_turn):
    return types.SimpleNamespace(transcript=transcript, end_of_turn=end_of_turn)


def test_human_turn_finalizes_on_end_of_turn():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("hello", False))
    r.turn(_turn("hello world", True))
    text = out.getvalue()
    assert "hello world" in text
    assert text.endswith("\n")


def test_json_mode_emits_ndjson_events():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.begin(types.SimpleNamespace(id="sess_1"))
    r.turn(_turn("hi", True))
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0] == {"type": "begin", "id": "sess_1"}
    assert lines[1] == {"type": "turn", "transcript": "hi", "end_of_turn": True}


def test_human_begin_prints_notice():
    out = io.StringIO()
    StreamRenderer(json_mode=False, out=out).begin(types.SimpleNamespace(id="x"))
    assert "Ctrl-C" in out.getvalue()


def test_human_shorter_turn_leaves_no_trailing_padding():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("hello world", False))  # long partial
    r.turn(_turn("hi", True))  # shorter, finalized
    # No leftover characters from the longer partial; finalized line ends clean.
    assert out.getvalue().endswith("hi\n")
    assert "hello world\n" not in out.getvalue()


def test_termination_json_emits_duration():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.termination(types.SimpleNamespace(audio_duration_seconds=12.5))
    import json as _json

    assert _json.loads(out.getvalue()) == {
        "type": "termination",
        "audio_duration_seconds": 12.5,
    }


def test_close_finalizes_open_partial_line():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("partial", False))  # no end_of_turn -> line left open
    r.close()
    assert out.getvalue().endswith("\n")


def test_close_is_noop_when_line_already_finalized():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("done", True))  # finalized with newline
    before = out.getvalue()
    r.close()
    assert out.getvalue() == before  # no extra newline


def test_close_is_noop_in_json_mode():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.turn(_turn("hi", False))
    before = out.getvalue()
    r.close()
    assert out.getvalue() == before


def test_close_swallows_non_pipe_errors():
    class FlakyOut:
        def write(self, _text):
            raise OSError("transient write error")

        def flush(self):
            pass

    r = StreamRenderer(json_mode=False, out=FlakyOut())
    r._line_open = True  # force the finalize path
    r.close()  # non-pipe errors are non-fatal


def test_llm_json_emits_event():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.llm("the summary")
    assert json.loads(out.getvalue()) == {"type": "llm", "content": "the summary"}


def test_llm_human_prints_content():
    out = io.StringIO()
    r = StreamRenderer(json_mode=False, out=out)
    r.turn(_turn("partial", False))  # open a partial line first
    r.llm("a tidy summary")
    text = out.getvalue()
    assert "a tidy summary" in text
    assert text.endswith("\n")


def test_llm_ignores_empty_content():
    out = io.StringIO()
    r = StreamRenderer(json_mode=True, out=out)
    r.llm("")  # nothing to show
    assert out.getvalue() == ""


def test_close_propagates_broken_pipe():
    import pytest

    class BrokenOut:
        def write(self, _text):
            raise BrokenPipeError("downstream closed")

        def flush(self):
            pass

    r = StreamRenderer(json_mode=False, out=BrokenOut())
    r._line_open = True  # force the finalize path
    with pytest.raises(BrokenPipeError):  # propagates so the command stops cleanly
        r.close()
