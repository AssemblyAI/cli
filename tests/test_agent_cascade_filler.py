"""Spoken-filler + planning-discard tests for `assembly live`.

While a tool runs the cascade speaks a short filler (rotated per tool, once per turn) so a
hands-free turn isn't dead air, and the deep agent's verbose planning between tool calls is held
unspoken so only the final answer is read aloud. Split into its own module (the brain and reply
suites stay under the 500-line file-length gate); driven against the shared cascade fakes — no
sockets, mic, or speaker.
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk

from aai_cli.agent_cascade import brain, weather_tool
from aai_cli.agent_cascade.brain import SpeechDelta, ToolNotice
from aai_cli.core.errors import APIError
from tests._cascade_fakes import make_session
from tests.test_agent_cascade_streamer import _collect, _MessageStreamGraph

# --- brain: the per-tool filler table + the carrier ToolNotice ---------------


def test_tool_fillers_maps_known_tools_and_falls_back_to_generic():
    # Each known tool carries its own spoken filler variants; an unknown/MCP tool falls back to
    # the generic tuple (mirrors how _tool_label falls back to "Using {name}").
    assert brain._tool_fillers(brain.WEB_SEARCH_TOOL_NAME) == (
        "Let me look that up.",
        "Searching now.",
        "One moment, checking the web.",
    )
    assert brain._tool_fillers(weather_tool.WEATHER_TOOL_NAME) == (
        "Let me check the weather.",
        "Checking the forecast now.",
    )
    assert brain._tool_fillers("totally_unknown_tool") == brain._GENERIC_FILLERS
    # The engine rotates fillers[index % len(fillers)], so an empty fallback would divide by zero.
    assert brain._GENERIC_FILLERS


def test_streamer_tool_notice_carries_the_tools_fillers():
    # The notice carries the tool's filler variants (not a pre-chosen one) so the engine owns
    # rotation; here the weather tool's tuple rides along with the affordance label.
    call_chunk = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {"name": weather_tool.WEATHER_TOOL_NAME, "args": "", "id": "c1", "index": 0}
        ],
    )
    graph = _MessageStreamGraph([(call_chunk, {})])
    events = _collect(graph, [{"role": "user", "content": "weather?"}])
    notices = [e for e in events if isinstance(e, brain.ToolNotice)]
    assert notices[0].fillers == brain._tool_fillers(weather_tool.WEATHER_TOOL_NAME)


# --- engine: speaking the filler + discarding interim planning ---------------


def test_generate_reply_speaks_a_filler_on_the_first_tool_call():
    # While a tool runs the agent says a spoken filler so a hands-free turn isn't dead air; it is
    # synthesized BEFORE the answer clauses (which only land after the tool returns).
    spoken = []

    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.", "Checking now."))
        yield SpeechDelta("It is sunny today.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["Let me check the weather.", "It is sunny today."]


def test_generate_reply_discards_planning_text_emitted_between_tool_calls():
    # A deepagents turn interleaves verbose planning ("SESSION INTENT / NEXT STEPS") with tool
    # calls; only the FINAL answer (the text after the last tool call) is spoken and recorded —
    # the intermediate planning is held unspoken and discarded at the next tool call.
    spoken = []

    def stream(messages):
        yield ToolNotice("Searching the web", ("Searching now.",))
        yield SpeechDelta("SESSION INTENT: the user asked for the news. ")  # planning, post-tool
        yield SpeechDelta("NEXT STEPS: read a few more pages. ")  # more planning
        yield ToolNotice("Reading the page", ("Reading now.",))  # discards the buffered planning
        yield SpeechDelta("Here is the news: AI is booming.")  # the real answer, after last tool

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["Searching now.", "Here is the news: AI is booming."]
    assert session.history[-1] == {
        "role": "assistant",
        "content": "Here is the news: AI is booming.",
    }


def test_generate_reply_speaks_only_one_filler_per_turn():
    # Only the first tool call of a turn speaks; chained tool calls stay silent so a multi-tool
    # turn doesn't get chatty.
    spoken = []

    def stream(messages):
        yield ToolNotice("Searching the web", ("look one", "look two"))
        yield ToolNotice("Reading the page", ("read one", "read two"))
        yield SpeechDelta("Done.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == ["look one", "Done."]  # the second tool call spoke no filler


def test_generate_reply_rotates_fillers_across_turns():
    # A per-session counter rotates variants deterministically, so the same tool across three
    # turns says each variant in turn rather than repeating one phrase.
    spoken = []
    variants = ("first.", "second.", "third.")

    def stream(messages):
        yield ToolNotice("Checking the weather", variants)
        yield SpeechDelta("ok.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    session._generate_reply()
    session._generate_reply()
    assert [s for s in spoken if s != "ok."] == ["first.", "second.", "third."]


def test_generate_reply_does_not_record_the_filler_in_history():
    # The filler is conversational glue, not part of the answer, so history stays a clean
    # alternating record of the real reply only.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("It is sunny today.")

    session, _renderer, _player = make_session(
        stream_reply=stream, synthesize=lambda text, sink: sink(b"")
    )
    session._generate_reply()
    assert session.history[-1] == {"role": "assistant", "content": "It is sunny today."}


def test_generate_reply_marks_started_when_a_turn_opens_with_a_tool_call():
    # The filler is the start of audible output, so _speaking is set and reply_started fires
    # before it's synthesized (so the voice bar shows speaking and a barge-in mid-filler is caught).
    observed = []

    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check.",))
        yield SpeechDelta("ok.")

    session, renderer, _player = make_session(stream_reply=stream)
    session.deps.synthesize = lambda text, sink: (
        observed.append((text, session._speaking.is_set())) or sink(b"")
    )
    session._generate_reply()
    assert ("reply_started",) in renderer.calls
    assert observed[0] == ("Let me check.", True)  # speaking set before the filler is synthesized


def test_barge_in_during_a_filler_drops_its_remaining_audio():
    # A barge-in mid-filler sets _stop; _feed must drop the in-flight frames just like a clause,
    # so nothing of the filler keeps playing after the user talks over it.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("answer")

    session, _renderer, player = make_session(stream_reply=stream)

    def synthesize(text, sink):
        session._stop.set()  # the user barges in mid-filler
        sink(b"frame")

    session.deps.synthesize = synthesize
    session._generate_reply()
    assert player.enqueued == []  # _feed dropped the frame once _stop was set


def test_generate_reply_aborts_when_the_filler_fails_to_synthesize():
    # A filler that can't be synthesized is the same failure mode as a clause that can't: the
    # error is recorded and the turn ends cleanly without speaking the answer.
    def stream(messages):
        yield ToolNotice("Checking the weather", ("Let me check the weather.",))
        yield SpeechDelta("the answer")

    spoken = []

    def synthesize(text, sink):
        if text == "the answer":
            spoken.append(text)
            return
        raise APIError("tts boom")

    session, _renderer, _player = make_session(stream_reply=stream, synthesize=synthesize)
    session._generate_reply()
    assert spoken == []  # the answer was never reached
    assert session.error is not None
    assert "tts boom" in session.error.message


def test_generate_reply_speaks_a_generic_filler_for_an_unknown_tool():
    # A tool with no entry in _TOOL_FILLERS still speaks — the notice carries the generic
    # fallback tuple (mirrors _tool_label's "Using {name}" fallback).
    spoken = []

    def stream(messages):
        yield ToolNotice("Using mcp_thing", brain._tool_fillers("mcp_thing"))
        yield SpeechDelta("done.")

    session, _renderer, _player = make_session(
        stream_reply=stream,
        synthesize=lambda text, sink: spoken.append(text) or sink(b""),
    )
    session._generate_reply()
    assert spoken == [brain._GENERIC_FILLERS[0], "done."]
