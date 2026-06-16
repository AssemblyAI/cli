"""Pure text helpers for the cascade: sentence splitting and history trimming.

Kept Rich-free and dependency-light so the orchestration logic in ``engine`` can
be unit-tested without any I/O.
"""

from __future__ import annotations

# A reply is spoken sentence-by-sentence so the first audio plays before the whole
# answer is synthesized; a sentence ends at one of these terminators.
_TERMINATORS = ".!?"


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each ending in ``.``/``!``/``?``.

    A trailing fragment with no terminal punctuation is kept as a final sentence,
    so no text is ever dropped; empty/whitespace-only pieces are discarded.
    """
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in _TERMINATORS:
            # The slice always includes the terminator at ``index``, so it is never
            # blank after stripping the inter-sentence whitespace.
            sentences.append(text[start : index + 1].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def trim_history[T](history: list[T], max_messages: int) -> None:
    """Cap ``history`` to its most recent ``max_messages`` entries, in place.

    A sliding window over the conversation so an unbounded chat doesn't grow the
    context (and the per-turn token cost) without limit.
    """
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
