"""DER scoring (`aai_cli.der`): pyannote-backed alignment, collar, pooling."""

import dataclasses

import pytest

from aai_cli import der


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_turns_and_scores_are_immutable_values():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(der.Turn(speaker="a", start=0.0, end=1.0), "speaker", "b")
    score = der.DerScore(missed=0.0, false_alarm=0.0, confusion=0.0, total=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(score, "total", 2.0)


REF = [
    der.Turn(speaker="alice", start=0.0, end=10.0),
    der.Turn(speaker="bob", start=10.0, end=20.0),
]
HYP = [der.Turn(speaker="A", start=0.0, end=12.0), der.Turn(speaker="B", start=12.0, end=20.0)]


def test_optimal_speaker_mapping_scores_only_the_overlap_error():
    # alice↔A and bob↔B map optimally; the 10-12s stretch is confusion.
    score = der.score(REF, HYP)
    assert score == der.DerScore(missed=0.0, false_alarm=0.0, confusion=2.0, total=20.0)
    assert score.der == 0.1


def test_collar_forgives_boundary_slop():
    # A 4s collar swallows the 10-12s confusion around the 10s boundary.
    score = der.score(REF, HYP, collar=4.0)
    assert score == der.DerScore(missed=0.0, false_alarm=0.0, confusion=0.0, total=12.0)
    assert score.der == 0.0


def test_empty_hypothesis_is_all_missed_detection():
    score = der.score(REF, [])
    assert score == der.DerScore(missed=20.0, false_alarm=0.0, confusion=0.0, total=20.0)
    assert score.der == 1.0


def test_hypothesis_speech_beyond_the_reference_is_false_alarm():
    # The evaluation extent covers the hypothesis too — speech past the
    # reference's end counts as false alarm rather than being cropped away.
    score = der.score(
        [der.Turn(speaker="a", start=0.0, end=10.0)],
        [der.Turn(speaker="A", start=0.0, end=15.0)],
    )
    assert score == der.DerScore(missed=0.0, false_alarm=5.0, confusion=0.0, total=10.0)
    assert score.der == 0.5


def test_overlapping_turns_by_one_speaker_keep_their_own_tracks():
    turns = [der.Turn(speaker="a", start=0.0, end=10.0), der.Turn(speaker="a", start=5.0, end=15.0)]
    assert der.score(turns, turns).der == 0.0


def test_pooled_sums_components():
    total = der.pooled(
        [
            der.DerScore(missed=1.0, false_alarm=2.0, confusion=3.0, total=10.0),
            der.DerScore(missed=0.0, false_alarm=0.0, confusion=0.0, total=10.0),
        ]
    )
    assert total == der.DerScore(missed=1.0, false_alarm=2.0, confusion=3.0, total=20.0)
    assert total.der == 0.3
