"""Tests for the cascade's pure text helpers."""

from __future__ import annotations

from aai_cli.agent_cascade.text import split_sentences, trim_history


def test_split_sentences_breaks_on_terminators():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_split_sentences_keeps_unterminated_tail():
    assert split_sentences("Done. And more") == ["Done.", "And more"]


def test_split_sentences_strips_whitespace_and_drops_empties():
    assert split_sentences("  Hi.   ") == ["Hi."]


def test_split_sentences_empty_string_is_empty_list():
    assert split_sentences("") == []


def test_split_sentences_terminator_followed_by_space_ends_a_sentence():
    # A terminator only closes the chunk when it ends the text or is followed by space.
    assert split_sentences(" . ") == ["."]
    assert split_sentences("Hi . Bye .") == ["Hi .", "Bye ."]


def test_split_sentences_keeps_decimals_and_abbreviations_intact():
    # A '.' wedged between non-space characters is not a sentence boundary, so a
    # number ("$3.50") or abbreviation stays one piece instead of fragmenting TTS.
    assert split_sentences("It costs $3.50 today.") == ["It costs $3.50 today."]
    assert split_sentences("Total 12.5") == ["Total 12.5"]


def test_split_sentences_does_not_split_stacked_terminators():
    # Ellipsis and "?!" are followed by non-space chars (or each other), so they
    # don't each spawn a separate sentence.
    assert split_sentences("...") == ["..."]
    assert split_sentences("Wait...what?!") == ["Wait...what?!"]


def test_trim_history_drops_oldest_beyond_limit():
    history = [{"role": "user", "content": str(i)} for i in range(5)]
    trim_history(history, 3)
    assert [item["content"] for item in history] == ["2", "3", "4"]


def test_trim_history_leaves_short_history_untouched():
    history = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    trim_history(history, 3)
    assert len(history) == 2


def test_trim_history_at_limit_is_untouched():
    history = [{"role": "user", "content": str(i)} for i in range(3)]
    trim_history(history, 3)
    assert len(history) == 3
