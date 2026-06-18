"""Pure text helpers for streaming TTS: sentence splitting and chunking.

PocketTTS (the streaming-TTS model behind ``assembly speak``) is fed incrementally —
a whole document in a single ``Generate`` frame stalls the server. ``chunk_text`` breaks
the input into sentence-aligned chunks small enough to synthesize one connection at a
time (see ``tts.session.synthesize_chunked``). Kept Rich-free and dependency-light so it
is trivially unit-testable.
"""

from __future__ import annotations

# A sentence ends at one of these terminators (mirrors the agent-cascade splitter).
_TERMINATORS = ".!?"

# Conservative upper bound on the characters in a single Generate frame. PocketTTS is a
# streaming model with a bounded context; everywhere else in the codebase it is fed one
# sentence at a time. Sentences are packed up to this budget to keep the connection count
# down on a long page while keeping each frame comfortably small.
_MAX_CHUNK_CHARS = 500  # pragma: no mutate -- a +-1 char budget is immaterial


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each ending in ``.``/``!``/``?``.

    A terminator ends a sentence only when it is the last character or is followed by
    whitespace — so a ``.`` inside a number ("$3.50") or stacked terminators ("..."/"?!")
    don't fragment one spoken sentence. A trailing fragment with no terminal punctuation
    is kept, so no text is ever dropped; empty/whitespace-only pieces are discarded.
    """
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in _TERMINATORS and (index + 1 == len(text) or text[index + 1].isspace()):
            # The two `+ 1`s below are equivalent under mutation: a confirmed boundary means
            # index+1 is whitespace or end-of-text, so widening the slice / advancing start by
            # one extra char only ever spans whitespace that .strip() removes.
            sentences.append(text[start : index + 1].strip())  # pragma: no mutate
            start = index + 1  # pragma: no mutate
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _bounded(sentence: str, max_chars: int) -> list[str]:
    """Slice ``sentence`` into ``<= max_chars`` pieces. A sentence within the budget comes
    back as a single piece (the one slice covers it); an over-long one (e.g. a PDF blob
    with no sentence terminators) is split so no single Generate frame can stall the
    server."""
    return [sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars)]


def chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into sentence-aligned chunks, each ``<= max_chars``.

    Sentences are packed greedily so short ones share a chunk (and thus one connection);
    packing never breaks mid-sentence unless a single sentence exceeds the budget, in
    which case that sentence alone is sliced. Whitespace-only input yields no chunks.
    """
    chunks: list[str] = []
    current = ""
    for sentence in split_sentences(text):
        for piece in _bounded(sentence, max_chars):
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = f"{current} {piece}"
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks
