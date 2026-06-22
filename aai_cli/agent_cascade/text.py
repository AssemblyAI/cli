"""Pure text helpers for the cascade: sentence splitting and history trimming.

Kept Rich-free and dependency-light so the orchestration logic in ``engine`` can
be unit-tested without any I/O.
"""

from __future__ import annotations

# A reply is spoken sentence-by-sentence so the first audio plays before the whole
# answer is synthesized; a sentence ends at one of these terminators.
_TERMINATORS = ".!?"

# Soft clause separators: a comma/semicolon/colon ends a *speakable* chunk too, but only
# once the pending clause is long enough (see pop_clauses) — flushing "Yes," on its own
# makes choppy TTS. Hard terminators (_TERMINATORS) always end a clause.
_SOFT_SEPARATORS = ",;:"


def _is_boundary(text: str, index: int) -> bool:
    """True when the char at ``index`` ends a clause: a terminator/separator that is the
    last char or is followed by whitespace (so a '.' inside "$3.50" never splits)."""
    return index + 1 == len(text) or text[index + 1].isspace()


def pop_clauses(buffer: str, *, min_chars: int) -> tuple[list[str], str]:
    """Pull complete speakable clauses off the front of ``buffer`` for incremental TTS.

    A hard terminator (``.``/``!``/``?``) followed by whitespace (or end-of-buffer) always
    ends a clause; a soft separator (``,``/``;``/``:``) ends one only when the clause built
    since the last boundary is at least ``min_chars`` long, so a tiny fragment ("Yes,")
    isn't synthesized on its own. Returns the flushed clauses (each stripped, never blank)
    and the still-incomplete remainder to keep buffering. The caller flushes the final tail
    at end-of-stream.
    """
    clauses: list[str] = []
    start = 0
    for index, char in enumerate(buffer):
        is_hard = char in _TERMINATORS
        is_soft = char in _SOFT_SEPARATORS
        if not (is_hard or is_soft) or not _is_boundary(buffer, index):
            continue
        clause = buffer[start : index + 1].strip()
        if is_soft and len(clause) < min_chars:
            continue  # too short to speak on its own — keep accumulating
        if clause:
            clauses.append(clause)
        start = index + 1
    return clauses, buffer[start:]


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, each ending in ``.``/``!``/``?``.

    A terminator ends a sentence only when it is the last character or is followed by
    whitespace — so a ``.`` inside a number ("$3.50") or stacked terminators ("..."/"?!")
    don't fragment one spoken sentence into several TTS calls (which both clips audio
    mid-number and writes a space-mangled copy back into the LLM history). A trailing
    fragment with no terminal punctuation is kept, so no text is ever dropped;
    empty/whitespace-only pieces are discarded.
    """
    sentences: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char in _TERMINATORS and (index + 1 == len(text) or text[index + 1].isspace()):
            # Boundary confirmed (end-of-text or a following space); the slice includes
            # the terminator, so it is never blank after stripping leading whitespace.
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
