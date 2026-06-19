import io
import json

import pytest

from aai_cli.agent.render import AgentRenderer
from aai_cli.ui import theme


def _json_lines(buf: io.StringIO):
    return [json.loads(x) for x in buf.getvalue().splitlines() if x.strip()]


def _human(width=80, color_system=None):
    """A human-mode renderer writing to a forced-terminal themed console buffer."""
    buf = io.StringIO()
    console = theme.make_console(
        file=buf, force_terminal=True, width=width, color_system=color_system
    )
    return AgentRenderer(json_mode=False, out=buf, console=console), buf


# --- json mode (unchanged) -------------------------------------------------
def test_json_emits_user_and_agent_events():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.connected()
    r.user_final("hello there")
    r.agent_transcript("hi back", interrupted=False)
    lines = _json_lines(buf)
    assert {"type": "session.ready"} in lines
    assert {"type": "transcript.user", "text": "hello there"} in lines
    assert {"type": "transcript.agent", "text": "hi back", "interrupted": False} in lines


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


def test_json_stopped_is_silent():
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).stopped()
    assert buf.getvalue() == ""


def test_json_close_is_silent():
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).close()
    assert buf.getvalue() == ""


def test_json_user_partial_emits_delta():
    buf = io.StringIO()
    r = AgentRenderer(json_mode=True, out=buf)
    r.user_partial("typing…")
    assert _json_lines(buf) == [{"type": "transcript.user.delta", "text": "typing…"}]


def test_json_emit_propagates_broken_pipe():
    class BrokenOut:
        def write(self, _text):
            raise BrokenPipeError("downstream closed")

        def flush(self):
            pass

    r = AgentRenderer(json_mode=True, out=BrokenOut())
    with pytest.raises(BrokenPipeError):  # propagates so the command stops cleanly
        r.connected()


def test_json_emit_swallows_non_pipe_errors():
    class FlakyOut:
        def write(self, _text):
            raise OSError("transient write error")

        def flush(self):
            pass

    AgentRenderer(json_mode=True, out=FlakyOut()).connected()  # non-pipe errors are non-fatal


# --- text mode (plain stdout + stderr status) ------------------------------
def test_text_connected_status_goes_to_stderr():
    out, err = io.StringIO(), io.StringIO()
    r = AgentRenderer(json_mode=False, text_mode=True, out=out, err=err)
    r.connected()
    # The "start talking" prompt is a status notice -> stderr, keeping stdout clean
    # for the piped transcript.
    assert "start talking" in err.getvalue().lower()
    assert out.getvalue() == ""


def test_text_notice_goes_to_stderr():
    out, err = io.StringIO(), io.StringIO()
    r = AgentRenderer(json_mode=False, text_mode=True, out=out, err=err)
    r.notice("Half-duplex note.\n")
    assert "Half-duplex note." in err.getvalue()
    assert out.getvalue() == ""


# --- human mode (Rich) -----------------------------------------------------
def test_human_partial_then_final():
    r, buf = _human()
    r.user_partial("what is")
    r.user_final("what is the time")
    r.close()
    assert "what is the time" in buf.getvalue()


def test_human_agent_line_labeled():
    r, buf = _human()
    r.agent_transcript("the time is noon", interrupted=False)
    out = buf.getvalue()
    assert "agent: " in out
    assert "the time is noon" in out


def test_json_tool_call_emits_tool_use_event():
    buf = io.StringIO()
    AgentRenderer(json_mode=True, out=buf).tool_call("Searching the web")
    assert {"type": "tool.use", "label": "Searching the web"} in _json_lines(buf)


def test_text_tool_call_goes_to_stderr_not_stdout():
    # The tool affordance is status, so in piped text mode it stays off stdout (transcript-only).
    out, err = io.StringIO(), io.StringIO()
    AgentRenderer(json_mode=False, text_mode=True, out=out, err=err).tool_call("Searching the web")
    assert "Searching the web" in err.getvalue()
    assert out.getvalue() == ""


def test_human_tool_call_shows_inline_line():
    r, buf = _human()
    r.tool_call("Searching the web")
    assert "Searching the web" in buf.getvalue()


def test_human_close_commits_open_partial():
    r, buf = _human()
    r.user_partial("half a sentence")
    r.close()
    assert "half a sentence" in buf.getvalue()  # committed, not dropped


def test_human_notice_goes_to_stderr_not_stdout():
    # Human (default) mode is also piped sometimes (`assembly agent | head`); the notice
    # must land on stderr in every non-JSON mode so stdout carries only transcript.
    out, err = io.StringIO(), io.StringIO()
    console = theme.make_console(file=out, force_terminal=True, width=80)
    r = AgentRenderer(json_mode=False, out=out, err=err, console=console)
    r.notice("Half-duplex note.\n")
    assert "Half-duplex note." in err.getvalue()
    assert out.getvalue() == ""


def test_json_notice_is_suppressed():
    out, err = io.StringIO(), io.StringIO()
    r = AgentRenderer(json_mode=True, out=out, err=err)
    r.notice("Half-duplex note.\n")
    assert out.getvalue() == ""
    assert err.getvalue() == ""


def test_human_connected_and_stopped_announce():
    r, buf = _human()
    r.connected()
    r.stopped()
    out = buf.getvalue()
    assert "start talking" in out.lower()
    assert "Stopped." in out


def test_human_connected_silent_without_mic_input():
    # File-driven runs have no mic, so the "start talking" prompt is suppressed.
    buf = io.StringIO()
    console = theme.make_console(file=buf, force_terminal=True, width=80)
    r = AgentRenderer(json_mode=False, mic_input=False, out=buf, console=console)
    r.connected()
    assert "start talking" not in buf.getvalue().lower()


def test_json_connected_still_emits_ready_without_mic_input():
    # The protocol event is independent of the human prompt.
    buf = io.StringIO()
    AgentRenderer(json_mode=True, mic_input=False, out=buf).connected()
    assert {"type": "session.ready"} in _json_lines(buf)


def test_human_agent_label_is_colored():
    r, buf = _human(color_system="truecolor")
    r.agent_transcript("the time is noon", interrupted=False)
    out = buf.getvalue()
    assert "agent: " in out
    assert "the time is noon" in out
    assert "\x1b[" in out  # label styling emits ANSI


def test_human_you_label_is_colored():
    r, buf = _human(color_system="truecolor")
    r.user_final("what is the time")
    r.close()
    out = buf.getvalue()
    assert "you: " in out
    assert "what is the time" in out
    assert "\x1b[" in out


def _truecolor_sgr(style_name: str) -> str:
    """The foreground SGR a truecolor console emits for a named theme style."""
    color = theme.make_console(color_system="truecolor").get_style(style_name).color
    assert color is not None
    t = color.get_truecolor()
    return f"\x1b[38;2;{t.red};{t.green};{t.blue}m"


def test_human_whole_turn_is_tinted_in_speaker_color():
    # The whole line — label *and* body — is wrapped in the speaker's color, and the two
    # speakers render in different colors, so a glance separates "you" from "agent".
    r, buf = _human(color_system="truecolor")
    r.user_final("hello")
    r.agent_transcript("hi there", interrupted=False)
    r.close()
    out = buf.getvalue()
    you_sgr, agent_sgr = _truecolor_sgr("aai.you"), _truecolor_sgr("aai.agent")
    assert you_sgr != agent_sgr  # contrasting hues, not two shades of the same color
    # The body text rides inside the colored span (it follows the SGR, before any reset).
    assert f"{you_sgr}you: hello" in out
    assert f"{agent_sgr}agent: hi there" in out
