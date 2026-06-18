"""Tests for `CodeSession`'s dual-mode streaming and cooperative cancellation.

Split from `test_code_agent.py` (which drives the real graph) to keep each file under the
500-line gate. These exercise the streaming loop with lightweight fakes: the session renders
from per-super-step ``"values"`` snapshots and checks the cancel flag on the frequent
per-token ``"messages"`` deltas, so a long generation can be interrupted promptly.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from aai_cli.code_agent.events import AssistantText
from aai_cli.code_agent.session import CodeSession


class StreamingAgent:
    """A double exercising the dual-mode streaming path.

    Mirrors langgraph's ``stream_mode=["values", "messages"]`` contract: each scripted state
    snapshot is yielded tagged as ``("values", snapshot)``, optionally preceded by
    ``("messages", delta)`` per-token deltas (the fine-grained cancellation checkpoints).
    """

    def __init__(
        self, chunks: list[dict[str, object]], *, token_deltas: tuple[str, ...] = ()
    ) -> None:
        self._chunks = chunks
        self._token_deltas = token_deltas

    def stream(self, graph_input, config=None, *, stream_mode=("values", "messages")):
        del graph_input, config, stream_mode
        for delta in self._token_deltas:
            yield ("messages", delta)
        for chunk in self._chunks:
            yield ("values", chunk)

    def invoke(self, *a, **k):  # the streaming branch is taken, so invoke is never used
        raise AssertionError("a streaming agent must not be invoked")


def test_send_streams_each_step_and_cancel_stops_the_loop() -> None:
    # Three successive graph states (messages grow by one each step); a stream_mode="values"
    # graph yields exactly these snapshots, so the session must emit incrementally.
    chunks: list[dict[str, object]] = [
        {"messages": [HumanMessage("go")]},
        {"messages": [HumanMessage("go"), AIMessage("first")]},
        {"messages": [HumanMessage("go"), AIMessage("first"), AIMessage("second")]},
    ]
    seen: list[object] = []
    session = CodeSession(
        agent=StreamingAgent(chunks), sink=seen.append, approver=lambda n, a: True
    )

    def sink(event: object) -> None:
        seen.append(event)
        if isinstance(event, AssistantText) and event.text == "first":
            session.request_cancel()  # cancel mid-stream, before the "second" chunk is consumed

    session.sink = sink
    session.send("go")

    texts = [e.text for e in seen if isinstance(e, AssistantText)]
    # "first" streamed out as its step landed; the cancel then broke the loop, so the later
    # "second" step was never emitted — proving both incremental rendering and cancellation.
    assert texts == ["first"]


def test_cancel_within_a_step_breaks_on_a_token_delta() -> None:
    # A single model generation is one super-step, so a values-only loop can't break until the
    # whole reply lands. Streaming the per-token "messages" deltas alongside gives a frequent
    # cancel checkpoint: a Ctrl-C mid-generation breaks before the reply ("late") is ever
    # rendered. Modeled by an agent that requests cancel between two token deltas.
    seen: list[object] = []

    class TokenStreamAgent:
        session: CodeSession

        def stream(self, graph_input, config=None, *, stream_mode=("values", "messages")):
            del graph_input, config, stream_mode
            yield ("messages", "par")  # first token arrives — loop sees no cancel yet
            self.session.request_cancel()  # user hits Ctrl-C mid-generation
            yield ("messages", "tial")  # next token: the loop's top-of-iteration check breaks
            yield ("values", {"messages": [AIMessage("late")]})  # must never be rendered

        def invoke(self, *a, **k):
            raise AssertionError("a streaming agent must not be invoked")

    agent = TokenStreamAgent()
    session = CodeSession(agent=agent, sink=seen.append, approver=lambda n, a: True)
    agent.session = session
    session.send("go")

    texts = [e.text for e in seen if isinstance(e, AssistantText)]
    assert texts == []  # the post-cancel "late" reply was dropped, not rendered


def test_only_values_chunks_are_rendered_not_messages_deltas() -> None:
    # The dual-mode stream tags each yield by mode; only "values" snapshots are rendered (the
    # "messages" deltas exist purely as cancel checkpoints). A messages delta that happens to
    # be a dict must NOT be emitted — guards the `mode == "values" and ...` guard against an
    # `and`->`or` slip that would render it.
    seen: list[object] = []

    class DualModeAgent:
        def stream(self, graph_input, config=None, *, stream_mode=("values", "messages")):
            del graph_input, config, stream_mode
            yield ("messages", {"messages": [AIMessage("ghost")]})  # dict, but messages-mode
            yield ("values", {"messages": [AIMessage("real")]})

        def invoke(self, *a, **k):
            raise AssertionError("a streaming agent must not be invoked")

    session = CodeSession(agent=DualModeAgent(), sink=seen.append, approver=lambda n, a: True)
    session.send("go")

    texts = [e.text for e in seen if isinstance(e, AssistantText)]
    assert texts == ["real"]  # the messages-mode dict ("ghost") was not rendered
