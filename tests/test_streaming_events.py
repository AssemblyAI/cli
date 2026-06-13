"""The typed streaming NDJSON events (`aai_cli.streaming.events`).

One canonical place that asserts each event's `type` discriminator and payload,
the omit-when-absent rule for `source`/`speaker`, and the closed/frozen model
guarantees the renderer relies on.
"""

import pytest
from pydantic import ValidationError

from aai_cli.streaming import events


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        # source is omitted when absent, but id stays present even when null.
        (events.Begin(session_id="sess_1"), {"type": "begin", "id": "sess_1"}),
        (events.Begin(session_id=None), {"type": "begin", "id": None}),
        (
            events.Begin(session_id="sess_1", source="system"),
            {"type": "begin", "id": "sess_1", "source": "system"},
        ),
        # speaker and source both drop out when undiarized / single-stream.
        (
            events.Turn(transcript="hi", end_of_turn=True),
            {"type": "turn", "transcript": "hi", "end_of_turn": True},
        ),
        (
            events.Turn(transcript="hi", end_of_turn=True, speaker="A", source="system"),
            {
                "type": "turn",
                "transcript": "hi",
                "end_of_turn": True,
                "speaker": "A",
                "source": "system",
            },
        ),
        # audio_duration_seconds stays present even when null.
        (
            events.Termination(audio_duration_seconds=12.5),
            {"type": "termination", "audio_duration_seconds": 12.5},
        ),
        (
            events.Termination(audio_duration_seconds=None),
            {"type": "termination", "audio_duration_seconds": None},
        ),
    ],
)
def test_wire_record(event: events.Event, expected: dict[str, object]):
    assert event.wire() == expected


def test_events_are_frozen():
    # An emitted event is immutable: the stream record can't be mutated after the fact.
    event = events.Turn(transcript="hi", end_of_turn=True)
    with pytest.raises(ValidationError):
        event.transcript = "tampered"


def test_events_reject_unknown_fields():
    # extra="forbid": a stray key is a programming error, not a silent passenger.
    with pytest.raises(ValidationError):
        events.Turn.model_validate({"transcript": "hi", "end_of_turn": True, "bogus": 1})
