"""Typed Voice Agent NDJSON events.

The ``assembly agent --json`` stream is a public contract: every line carries a
``type`` discriminator (see docs/cli-reference.md) and a small, fixed payload.
Modelling each event as a Pydantic class — rather than hand-building
``{"type": …}`` dicts the type checker only sees as ``dict[str, str]`` — pins the
``type`` literal and the payload fields at type-check time, so a renamed key or a
mistyped ``type`` value fails before it can drift onto the wire. ``model_dump()``
is the serialized form the renderer emits.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Event(BaseModel):
    """Base for Voice Agent events: a closed, frozen wire model.

    ``extra="forbid"`` keeps a stray field from silently riding along on the
    stream, and ``frozen=True`` makes an emitted event immutable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class SessionReady(_Event):
    """The agent session connected and is ready for audio."""

    type: Literal["session.ready"] = "session.ready"


class UserDelta(_Event):
    """An interim (partial) user transcript."""

    type: Literal["transcript.user.delta"] = "transcript.user.delta"
    text: str


class UserFinal(_Event):
    """A finalized user transcript turn."""

    type: Literal["transcript.user"] = "transcript.user"
    text: str


class ReplyStarted(_Event):
    """The agent began generating a reply."""

    type: Literal["reply.started"] = "reply.started"


class AgentTranscript(_Event):
    """The agent's reply transcript (``interrupted`` when the user barged in)."""

    type: Literal["transcript.agent"] = "transcript.agent"
    text: str
    interrupted: bool


class ReplyDone(_Event):
    """The agent finished, or was interrupted out of, a reply."""

    type: Literal["reply.done"] = "reply.done"
    interrupted: bool


Event = SessionReady | UserDelta | UserFinal | ReplyStarted | AgentTranscript | ReplyDone
