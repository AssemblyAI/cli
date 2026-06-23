"""The typed Voice Agent NDJSON events (`aai_cli.agent.events`).

These pin the wire contract `assembly agent --json` emits: one canonical place
that asserts each event's `type` discriminator and payload, plus the closed/
frozen model guarantees the renderer relies on.
"""

import pytest
from pydantic import ValidationError

from aai_cli.agent import events


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (events.SessionReady(), {"type": "session.ready"}),
        (events.UserDelta(text="typing…"), {"type": "transcript.user.delta", "text": "typing…"}),
        (events.UserFinal(text="hello"), {"type": "transcript.user", "text": "hello"}),
        (events.ReplyStarted(), {"type": "reply.started"}),
        (
            events.ToolUse(label="Searching the web"),
            {"type": "tool.use", "label": "Searching the web"},
        ),
        (
            events.PlanUpdate(todos=(events.TodoItem(content="Book a flight", status="pending"),)),
            # model_dump keeps todos a tuple of nested dicts; it serializes to a JSON array on the
            # wire (asserted end-to-end in test_agent_render.test_json_todos_emits_plan_event).
            {"type": "plan", "todos": ({"content": "Book a flight", "status": "pending"},)},
        ),
        (
            events.AgentTranscript(text="hi back", interrupted=False),
            {"type": "transcript.agent", "text": "hi back", "interrupted": False},
        ),
        (events.ReplyDone(interrupted=True), {"type": "reply.done", "interrupted": True}),
    ],
)
def test_event_wire_shape(event: events.Event, expected: dict[str, object]):
    assert event.model_dump() == expected


def test_events_are_frozen():
    # An emitted event is immutable: the stream record can't be mutated after the fact.
    event = events.UserFinal(text="hello")
    with pytest.raises(ValidationError):
        event.text = "tampered"


def test_events_reject_unknown_fields():
    # extra="forbid": a stray key is a programming error, not a silent passenger.
    with pytest.raises(ValidationError):
        events.UserFinal.model_validate({"text": "hello", "bogus": 1})
