"""Tests for the cascade's pure text helpers (sentence/clause splitting)."""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade.text import pop_clauses, split_sentences


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


def test_pop_clauses_flushes_hard_terminators_and_keeps_tail():
    chunks, remainder = pop_clauses("One. Two! Three", min_chars=1)
    assert chunks == ["One.", "Two!"]
    assert remainder == " Three"  # no terminator yet -> stays buffered


def test_pop_clauses_flushes_soft_separator_only_past_min_chars():
    # The clause before the comma is long enough, so the comma ends a clause.
    chunks, remainder = pop_clauses("the weather today is, in fact ", min_chars=10)
    assert chunks == ["the weather today is,"]
    assert remainder == " in fact "


def test_pop_clauses_holds_short_soft_clause_to_avoid_choppy_tts():
    # "Yes," is shorter than min_chars, so it is NOT flushed on the comma.
    chunks, remainder = pop_clauses("Yes, it is sunny", min_chars=10)
    assert chunks == []
    assert remainder == "Yes, it is sunny"


def test_pop_clauses_does_not_fragment_a_decimal_or_stacked_terminators():
    # A '.' inside $3.50 (no following space) and stacked '...'/'?!' are not boundaries.
    chunks, remainder = pop_clauses("It costs $3.50 total... ", min_chars=1)
    assert chunks == ["It costs $3.50 total..."]
    assert remainder == " "


def test_pop_clauses_returns_nothing_for_an_unterminated_buffer():
    chunks, remainder = pop_clauses("still going", min_chars=1)
    assert chunks == []
    assert remainder == "still going"


def test_pop_clauses_strips_whitespace_from_each_flushed_clause():
    # Trailing space after the last "." so it's a confirmed boundary (a terminator at end-of-buffer
    # is held, not flushed — see test_pop_clauses_holds_terminator_at_end_of_buffer).
    chunks, _remainder = pop_clauses("  Hi there.  Next. ", min_chars=1)
    assert chunks == ["Hi there.", "Next."]


def test_pop_clauses_holds_terminator_at_end_of_buffer():
    # A terminator sitting at the current end of a streamed chunk may be mid-token ("$3." before
    # "50" arrives), so it is held in the remainder rather than split into its own clause.
    chunks, remainder = pop_clauses("It costs $3.", min_chars=1)
    assert chunks == []
    assert remainder == "It costs $3."
    # Once the rest streams in and a real boundary follows, the number is spoken whole.
    chunks, remainder = pop_clauses("It costs $3.50 today. ", min_chars=1)
    assert chunks == ["It costs $3.50 today."]
    assert remainder == " "


@pytest.mark.parametrize("min_chars", [1, 25])
def test_pop_clauses_flushes_hard_terminator_regardless_of_min_chars(min_chars):
    # min_chars only gates SOFT separators; a sentence terminator always flushes.
    chunks, remainder = pop_clauses("Hi. ", min_chars=min_chars)
    assert chunks == ["Hi."]
    assert remainder == " "
