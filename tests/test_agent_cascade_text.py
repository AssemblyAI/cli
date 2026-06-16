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


def test_split_sentences_each_terminator_ends_a_sentence():
    # Every terminator closes the current chunk, so consecutive ones each yield one.
    assert split_sentences("...") == [".", ".", "."]
    assert split_sentences(" . ") == ["."]


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
