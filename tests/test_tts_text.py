from __future__ import annotations

from aai_cli.tts import text


def test_chunk_text_packs_short_sentences_into_one_chunk():
    # Several short sentences well under the budget ride in a single chunk (one
    # connection) rather than one connection per sentence.
    out = text.chunk_text("One. Two. Three.", max_chars=100)
    assert out == ["One. Two. Three."]


def test_chunk_text_does_not_break_on_mid_number_period():
    # A "." inside "$3.50" is not a sentence boundary (no following whitespace), so the
    # whole sentence stays in one chunk.
    assert text.chunk_text("It costs $3.50 today.", max_chars=100) == ["It costs $3.50 today."]


def test_chunk_text_packs_two_sentences_exactly_at_the_budget():
    # "ab." + " " + "cd." == 7 chars: at a budget of 7 they pack into one chunk. Pins that
    # the sentence-terminator separator packs rather than splitting at the boundary.
    assert text.chunk_text("ab. cd.", max_chars=7) == ["ab. cd."]


def test_chunk_text_splits_two_sentences_one_over_the_budget():
    # The same two sentences need 7 chars joined; a budget of 6 can't hold both, so the
    # second rolls to its own chunk (the break lands on the sentence terminator, never
    # mid-word).
    assert text.chunk_text("ab. cd.", max_chars=6) == ["ab.", "cd."]


def test_chunk_text_slices_a_single_oversized_sentence():
    # A lone "sentence" longer than the budget (no terminators or spaces — the PDF blob
    # case) falls all the way to the bare-character separator so no single Generate frame
    # can blow past the server's input ceiling.
    out = text.chunk_text("abcdefghij", max_chars=4)
    assert out == ["abcd", "efgh", "ij"]
    assert all(len(piece) <= 4 for piece in out)


def test_chunk_text_empty_input_returns_no_chunks():
    assert text.chunk_text("   ") == []


def test_chunk_text_every_chunk_within_budget_for_a_long_paragraph():
    para = " ".join(f"Sentence number {n} here." for n in range(200))
    out = text.chunk_text(para, max_chars=120)
    assert len(out) > 1  # a long paragraph really is chunked
    assert all(len(piece) <= 120 for piece in out)
    # No text is lost (and none is duplicated by overlap): rejoining the chunks recovers
    # every word in order.
    assert out and " ".join(out).split() == para.split()
