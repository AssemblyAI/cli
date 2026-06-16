"""DER scoring (`aai_cli.core.der`): timeline tally, optimal speaker mapping, pooling."""

import dataclasses

import pytest

from aai_cli.core import der


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def seg(speaker: str, start: float, end: float) -> der.Segment:
    return der.Segment(speaker, start, end)


def test_score_is_an_immutable_value():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(der.Score(missed=0.0, false_alarm=0.0, confusion=0.0, speech=1.0), "missed", 1.0)


def test_errors_and_der_combine_the_components():
    score = der.Score(missed=1.0, false_alarm=2.0, confusion=3.0, speech=10.0)
    assert score.errors == 6.0
    assert score.der == 0.6


def test_identical_diarization_scores_zero():
    ref = [seg("A", 0, 10), seg("B", 10, 20)]
    score = der.score(ref, ref)
    assert score == der.Score(missed=0.0, false_alarm=0.0, confusion=0.0, speech=20.0)
    assert score.der == 0.0


def test_speaker_labels_are_mapped_optimally():
    # Same timing, relabelled and reversed: the optimal 1:1 mapping recovers it,
    # so a correct scorer reports no error despite none of the labels matching.
    ref = [seg("A", 0, 10), seg("B", 10, 20)]
    hyp = [seg("spk_1", 10, 20), seg("spk_0", 0, 10)]
    assert der.score(ref, hyp).der == 0.0


def test_missing_hypothesis_is_all_missed_speech():
    score = der.score([seg("A", 0, 10)], [])
    assert score == der.Score(missed=10.0, false_alarm=0.0, confusion=0.0, speech=10.0)
    assert score.der == 1.0


def test_hypothesis_speech_outside_the_reference_is_false_alarm():
    score = der.score([seg("A", 0, 10)], [seg("X", 0, 15)])
    assert score == der.Score(missed=0.0, false_alarm=5.0, confusion=0.0, speech=10.0)
    assert score.der == 0.5


def test_no_reference_speech_is_pure_false_alarm():
    # An empty reference: every hypothesis second is a false alarm and there is
    # no speech to map against (DER itself is undefined, but the tally holds).
    score = der.score([], [seg("X", 0, 10)])
    assert score == der.Score(missed=0.0, false_alarm=10.0, confusion=0.0, speech=0.0)


def test_one_hypothesis_speaker_split_across_two_references_is_confusion():
    # A single hypothesis speaker covers both reference speakers; the mapping can
    # only credit the one it overlaps most (B, 10s), the rest (A, 3s) is confusion.
    ref = [seg("A", 0, 3), seg("B", 3, 13)]
    hyp = [seg("X", 0, 13)]
    score = der.score(ref, hyp)
    assert score == der.Score(missed=0.0, false_alarm=0.0, confusion=3.0, speech=13.0)
    assert score.der == pytest.approx(3 / 13)


def test_one_reference_speaker_split_across_two_hypotheses_is_confusion():
    # The mirror case (more hypothesis than reference speakers): only one of the
    # two hypothesis speakers can be mapped to A, the other 5s is confusion.
    ref = [seg("A", 0, 10)]
    hyp = [seg("X", 0, 5), seg("Y", 5, 10)]
    score = der.score(ref, hyp)
    assert score == der.Score(missed=0.0, false_alarm=0.0, confusion=5.0, speech=10.0)
    assert score.der == 0.5


def test_optimal_mapping_beats_a_greedy_one():
    # Co-occurrence is (A,X)=10, (A,Y)=9, (B,X)=8. Greedy takes A↔X (10) and is
    # then stuck with B↔Y (0) for 10 correct; the optimal A↔Y + B↔X gives 17
    # correct, so confusion is 27-17=10, not 27-10=17. Pins the max-assignment.
    ref = [seg("A", 0, 19), seg("B", 19, 27)]
    hyp = [seg("X", 0, 10), seg("Y", 10, 19), seg("X", 19, 27)]
    score = der.score(ref, hyp)
    assert score == der.Score(missed=0.0, false_alarm=0.0, confusion=10.0, speech=27.0)
    assert score.der == pytest.approx(10 / 27)


def test_overlapping_reference_speakers_count_per_speaker():
    # [5,10) has two reference speakers talking at once, so it contributes 2x its
    # 5s of wall-clock to reference speech (15s wall-clock -> 20s speaker-time).
    ref = [seg("A", 0, 10), seg("B", 5, 15)]
    score = der.score(ref, ref)
    assert score == der.Score(missed=0.0, false_alarm=0.0, confusion=0.0, speech=20.0)


def test_pooled_sums_components_for_corpus_der():
    total = der.pooled(
        [
            der.Score(missed=1.0, false_alarm=2.0, confusion=3.0, speech=10.0),
            der.Score(missed=0.0, false_alarm=1.0, confusion=1.0, speech=30.0),
        ]
    )
    assert total == der.Score(missed=1.0, false_alarm=3.0, confusion=4.0, speech=40.0)
    assert total.der == 0.2
