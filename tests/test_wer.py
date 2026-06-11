"""WER scoring (`aai_cli.wer`): normalization, alignment counts, pooling."""

import dataclasses

import pytest

from aai_cli import wer


def _assign(obj, attribute, value):
    setattr(obj, attribute, value)


def test_score_is_an_immutable_value():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _assign(wer.Score(errors=0, words=1), "errors", 1)


def test_normalize_lowercases_and_strips_punctuation():
    assert wer.normalize_words("Hello, WORLD! It's fine.") == ["hello", "world", "its", "fine"]


def test_normalize_empty_and_punctuation_only_yield_no_words():
    assert wer.normalize_words("") == []
    assert wer.normalize_words("...!!!") == []


def test_identical_texts_score_zero():
    score = wer.score("the quick brown fox", "the quick brown fox")
    assert score == wer.Score(errors=0, words=4)
    assert score.wer == 0.0


def test_normalization_differences_do_not_count_as_errors():
    assert wer.score("Hello, world.", "hello world") == wer.Score(errors=0, words=2)


def test_single_substitution():
    score = wer.score("a b c d", "a x c d")
    assert score == wer.Score(errors=1, words=4)
    assert score.wer == 0.25


def test_single_deletion():
    assert wer.score("a b c", "a c") == wer.Score(errors=1, words=3)


def test_insertions_count_against_reference_length():
    # Two inserted words; the reference length (the denominator) stays 3, so
    # WER can exceed what a substitution-only alignment would give.
    score = wer.score("a b c", "x a b c y")
    assert score == wer.Score(errors=2, words=3)


def test_empty_hypothesis_is_all_deletions():
    score = wer.score("a b c", "")
    assert score == wer.Score(errors=3, words=3)
    assert score.wer == 1.0


def test_pooled_sums_errors_and_words():
    total = wer.pooled([wer.Score(errors=1, words=4), wer.Score(errors=2, words=6)])
    assert total == wer.Score(errors=3, words=10)
    assert total.wer == 0.3
