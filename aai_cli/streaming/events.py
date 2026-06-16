"""Typed streaming NDJSON events.

The ``assembly stream --json`` stream is a public contract: every line carries a
``type`` discriminator (see docs/cli-reference.md). Modelling each event as a
Pydantic class — rather than hand-building ``{"type": …}`` dicts the type checker
only sees as ``dict[str, object]`` — pins each event's ``type`` literal and
payload at type-check time.

Two presence rules the renderer relied on are preserved in ``wire()``: the
optional *annotations* ``source`` (parallel system/you streams) and ``speaker``
(``--speaker-labels`` diarization) drop out of the record when absent, while the
core payload (``id``, ``audio_duration_seconds``) stays present even when null.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Optional annotations omitted from the record when None; the core payload is kept.
_OMIT_WHEN_NONE = ("speaker", "source")


class _StreamEvent(BaseModel):
    """Base for streaming events: a closed, frozen wire model.

    ``extra="forbid"`` keeps a stray field off the stream and ``frozen=True``
    makes an emitted event immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    def wire(self) -> dict[str, object]:
        """The NDJSON record: optional annotations drop out when absent."""
        data: dict[str, object] = self.model_dump(by_alias=True)
        for key in _OMIT_WHEN_NONE:
            if data.get(key) is None:
                data.pop(key, None)
        return data


class Begin(_StreamEvent):
    """The session opened; ``id`` is the streaming session id."""

    type: Literal["begin"] = "begin"
    session_id: str | None = Field(serialization_alias="id")
    source: str | None = None


class Turn(_StreamEvent):
    """A turn transcript: interim while ``end_of_turn`` is False, finalized when True."""

    type: Literal["turn"] = "turn"
    transcript: str
    end_of_turn: bool
    speaker: str | None = None
    source: str | None = None


class Termination(_StreamEvent):
    """The session closed; ``audio_duration_seconds`` is the total audio processed."""

    type: Literal["termination"] = "termination"
    audio_duration_seconds: float | None
    source: str | None = None


class Source(_StreamEvent):
    """A ``--from-stdin`` batch advanced to its next audio source.

    Emitted once before each source's own ``begin``/``turn``/``termination`` events,
    so a consumer can segment the NDJSON stream by source. ``index`` is 1-based.
    """

    type: Literal["source"] = "source"
    source: str
    index: int
    total: int


Event = Begin | Turn | Termination | Source
