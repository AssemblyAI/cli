import io
import json

from assemblyai_cli.agent.render import AgentRenderer


def _json_lines(buf: io.StringIO):
    return [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]


def test_json_emits_user_and_agent_events():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.connected()
    r.user_final("hello there")
    r.agent_transcript("hi back", interrupted=False)
    lines = _json_lines(buf)
    assert {"type": "session.ready"} in lines
    assert {"type": "transcript.user", "text": "hello there"} in lines
    assert {
        "type": "transcript.agent",
        "text": "hi back",
        "interrupted": False,
    } in lines


def test_json_never_emits_audio_bytes():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.reply_started()
    r.reply_done(interrupted=True)
    text = buf.getvalue()
    assert "data" not in text  # no base64 audio leaks
    lines = _json_lines(buf)
    assert {"type": "reply.started"} in lines
    assert {"type": "reply.done", "interrupted": True} in lines


def test_human_partial_updates_in_place_then_finalizes():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.user_partial("what is")
    r.user_final("what is the time")
    out = buf.getvalue()
    assert "\r\x1b[K" in out  # cleared the line for the partial
    assert "what is the time" in out  # finalized text present
    assert out.endswith("\n")  # finalized line ends clean


def test_human_agent_line_labeled():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.agent_transcript("the time is noon", interrupted=False)
    out = buf.getvalue()
    assert out.startswith("agent: ")
    assert "the time is noon" in out


def test_close_finalizes_open_partial_line():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.user_partial("half a sen")
    r.close()
    assert buf.getvalue().endswith("\n")


def test_json_stopped_is_silent():
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).stopped()
    assert buf.getvalue() == ""


def test_json_close_is_silent():
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).close()
    assert buf.getvalue() == ""


def test_notice_writes_to_buffer_in_human_mode():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.notice("hi")
    assert buf.getvalue() == "hi"


def test_human_connected_and_stopped_announce():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=False, out=buf)
    r.connected()
    r.stopped()
    out = buf.getvalue()
    assert "start talking" in out.lower()
    assert "Stopped." in out


def test_json_user_partial_emits_delta():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.user_partial("typing…")
    assert _json_lines(buf) == [{"type": "transcript.user.delta", "text": "typing…"}]


def test_write_swallows_non_pipe_errors():
    class FlakyOut:
        def write(self, _text):
            raise OSError("transient write error")

        def flush(self):
            pass

    # Non-pipe write errors are non-fatal and must not raise.
    AgentRenderer(json_mode=False, out=FlakyOut()).notice("anything")


def test_write_propagates_broken_pipe():
    import pytest

    class BrokenOut:
        def write(self, _text):
            raise BrokenPipeError("downstream closed")

        def flush(self):
            pass

    # BrokenPipeError must propagate so the command can stop cleanly (`| head`).
    with pytest.raises(BrokenPipeError):
        AgentRenderer(json_mode=False, out=BrokenOut()).notice("x")
