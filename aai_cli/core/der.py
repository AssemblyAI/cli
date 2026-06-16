"""Diarization error rate (DER) scoring for `assembly eval`, dependency-free.

The companion to :mod:`aai_cli.core.wer`: where WER scores *what* was said, DER
scores *who* spoke when. It is the standard NIST/pyannote metric — the fraction
of reference speech time that is misattributed — computed here in plain Python so
adding diarization metrics costs no new dependency (``pyannote.metrics`` drags in
numpy/scipy/pandas; the lighter PyPI options still pull numpy or compile a C++
extension). No SDK, no Rich: the command layer owns all rendering.

Both reference and hypothesis are sequences of speaker-labelled :class:`Segment`s.
Speaker labels are arbitrary, so the speakers are optimally mapped one-to-one
before errors are counted (an exact search — diarization speaker counts are
small). The timeline is partitioned at every segment boundary into atomic
intervals; within each, the per-interval NIST tally (missed / false-alarm /
confusion) is summed, all weighted by reference *speaker*-time so overlapping
speech is counted once per concurrent speaker.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A stretch of speech (``start`` to ``end``, in seconds) attributed to one speaker."""

    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class Score:
    """Diarization error against ``speech`` seconds of reference speech.

    The three NIST components are kept separately (so a caller can show the
    breakdown) and ``errors`` is their sum; pooled across files for corpus DER.
    """

    missed: float
    false_alarm: float
    confusion: float
    speech: float

    @property
    def errors(self) -> float:
        """Total misattributed time: missed + false-alarm + speaker-confusion seconds."""
        return self.missed + self.false_alarm + self.confusion

    @property
    def der(self) -> float:
        """Diarization error rate: error time over reference speech time.

        The caller guarantees a reference with speech (an empty reference makes
        DER undefined, the same contract :class:`wer.Score` keeps for ``words``).
        """
        return self.errors / self.speech


def _boundaries(reference: Sequence[Segment], hypothesis: Sequence[Segment]) -> list[float]:
    """The sorted, de-duplicated segment endpoints that partition the timeline.

    Between two consecutive boundaries every segment is wholly present or wholly
    absent, so each atomic interval has a fixed set of active speakers.
    """
    return sorted({point for seg in (*reference, *hypothesis) for point in (seg.start, seg.end)})


def _active(segments: Sequence[Segment], start: float, end: float) -> set[str]:
    """The distinct speakers whose segment covers the atomic interval ``[start, end)``."""
    return {seg.speaker for seg in segments if seg.start <= start and seg.end >= end}


def _speakers(segments: Sequence[Segment]) -> list[str]:
    """The distinct speaker labels in ``segments``, in a deterministic order."""
    return sorted({seg.speaker for seg in segments})


def _weight(weights: list[list[float]], row: int, col: int) -> float:
    """The matrix weight at ``(row, col)``, or 0 outside it — the zero-padding
    that lets the larger side's unmatched speakers cost nothing."""
    if row < len(weights) and col < len(weights[0]):
        return weights[row][col]
    return 0.0


def _max_weight_assignment(weights: list[list[float]]) -> float:
    """The greatest total weight of a one-to-one row-to-column assignment.

    Exact search: the matrix is zero-padded to ``size x size`` and every
    permutation (speaker mapping) is tried. Diarization files have few speakers,
    so the factorial search stays cheap, and padding sidesteps the orientation
    branch a rectangular search would need.
    """
    size = max(len(weights), len(weights[0]))
    return max(
        sum(_weight(weights, row, col) for row, col in enumerate(perm))
        for perm in itertools.permutations(range(size))
    )


def _correct_time(
    reference: Sequence[Segment],
    hypothesis: Sequence[Segment],
    cooccurrence: dict[tuple[str, str], float],
) -> float:
    """Correctly attributed speech time under the optimal speaker mapping.

    ``cooccurrence[(ref, hyp)]`` is how long that reference and hypothesis
    speaker were concurrently active; the best one-to-one mapping maximises the
    matched total (an empty hypothesis maps to nothing, so it scores 0).
    """
    ref_speakers, hyp_speakers = _speakers(reference), _speakers(hypothesis)
    if not ref_speakers:
        return 0.0
    weights = [[cooccurrence.get((ref, hyp), 0.0) for hyp in hyp_speakers] for ref in ref_speakers]
    return _max_weight_assignment(weights)


def score(reference: Sequence[Segment], hypothesis: Sequence[Segment]) -> Score:
    """Score a hypothesis diarization against a reference.

    Walks the shared timeline once, tallying missed speech (reference speakers
    with no hypothesis speaker to cover them), false alarms (the reverse), and
    co-occurrence per speaker pair; speaker confusion is then the matched
    overlap that the optimal mapping could *not* account for.
    """
    cooccurrence: dict[tuple[str, str], float] = {}
    missed = false_alarm = matched = speech = 0.0
    boundaries = _boundaries(reference, hypothesis)
    for start, end in itertools.pairwise(boundaries):
        duration = end - start
        ref_active = _active(reference, start, end)
        hyp_active = _active(hypothesis, start, end)
        speech += duration * len(ref_active)
        missed += duration * max(0, len(ref_active) - len(hyp_active))
        false_alarm += duration * max(0, len(hyp_active) - len(ref_active))
        matched += duration * min(len(ref_active), len(hyp_active))
        for ref in ref_active:
            for hyp in hyp_active:
                cooccurrence[(ref, hyp)] = cooccurrence.get((ref, hyp), 0.0) + duration
    confusion = matched - _correct_time(reference, hypothesis, cooccurrence)
    return Score(missed=missed, false_alarm=false_alarm, confusion=confusion, speech=speech)


def pooled(scores: list[Score]) -> Score:
    """Corpus-level score: error and speech time summed across files (DER is then
    total error time over total reference time, not a mean of per-file rates)."""
    return Score(
        missed=sum(item.missed for item in scores),
        false_alarm=sum(item.false_alarm for item in scores),
        confusion=sum(item.confusion for item in scores),
        speech=sum(item.speech for item in scores),
    )
