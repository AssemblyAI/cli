"""Diarization error rate (DER) scoring for `assembly eval --speaker-labels`.

A thin shim over :mod:`pyannote.metrics` — the de-facto standard DER
implementation, including its optimal reference↔hypothesis speaker mapping —
so the alignment math is never re-derived here. pyannote drags numpy/scipy/
pandas along, so it is imported lazily inside :func:`score`; the dataclasses
here stay import-cheap for the command layer. No SDK, no Rich.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

if TYPE_CHECKING:
    from pyannote.core import Annotation

# pyannote types the metric call as a bare float; with ``detailed=True`` it
# actually returns the components dict — validate that shape instead of casting.
_COMPONENTS: TypeAdapter[dict[str, float]] = TypeAdapter(dict[str, float])


@dataclass(frozen=True)
class Turn:
    """One speaker turn: who spoke from ``start`` to ``end`` (seconds)."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class DerScore:
    """DER components in seconds of speech; pooled across files for corpus DER."""

    missed: float
    false_alarm: float
    confusion: float
    total: float

    @property
    def der(self) -> float:
        return (self.missed + self.false_alarm + self.confusion) / self.total


def _annotation(turns: list[Turn]) -> Annotation:
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    # The list index keys each track, so two overlapping turns by the same
    # speaker don't collapse into one.
    for track, turn in enumerate(turns):
        annotation[Segment(turn.start, turn.end), track] = turn.speaker
    return annotation


def score(reference: list[Turn], hypothesis: list[Turn], *, collar: float = 0.0) -> DerScore:
    """Score hypothesis speaker turns against the reference diarization.

    The caller guarantees a non-empty reference (the dataset loader rejects
    rows without speaker turns), so ``DerScore.der`` is always well-defined.
    ``collar`` forgives that many seconds around each reference turn boundary.
    """
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    metric = DiarizationErrorRate(collar=collar)
    # An explicit evaluation extent (audio start through the last turn either
    # side heard); without it pyannote warns about approximating the UEM.
    extent = max(turn.end for turn in [*reference, *hypothesis])
    components: object = metric(
        _annotation(reference),
        _annotation(hypothesis),
        uem=Timeline([Segment(0.0, extent)]),
        detailed=True,
    )
    detail = _COMPONENTS.validate_python(components)
    return DerScore(
        missed=detail["missed detection"],
        false_alarm=detail["false alarm"],
        confusion=detail["confusion"],
        total=detail["total"],
    )


def pooled(scores: list[DerScore]) -> DerScore:
    """Corpus-level score: error seconds over total reference speech seconds."""
    return DerScore(
        missed=sum(item.missed for item in scores),
        false_alarm=sum(item.false_alarm for item in scores),
        confusion=sum(item.confusion for item in scores),
        total=sum(item.total for item in scores),
    )
