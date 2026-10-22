"""Text chunking for streaming TTS, built on LangChain's text splitters.

Streaming TTS takes text in bounded requests — a single ``Generate`` frame holds at
most 2000 characters, and a whole document sent as one burst closes the session.
``chunk_text`` breaks the input into sentence-aligned chunks, each sent as its own
request on one session (see ``tts.session.synthesize_chunked``). The splitting is delegated to LangChain's
``RecursiveCharacterTextSplitter`` (https://docs.langchain.com/oss/python/integrations/splitters)
rather than a hand-rolled sentence scanner: it recurses through the separator list below,
preferring sentence boundaries and only falling back to words/characters when a single
sentence overflows the budget.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Conservative upper bound on the characters in a single Generate frame — well under the
# service's 2000-character limit. Sentences are packed up to this budget to keep the
# request count down on a long page while keeping each request comfortably small.
_MAX_CHUNK_CHARS = 500  # pragma: no mutate -- a +-1 char budget is immaterial

# Separators in descending priority: a sentence terminator followed by a space is the
# preferred break, then paragraph/line breaks, then a word boundary, and finally a bare
# character split for an over-long blob with none of the above (e.g. a PDF with no
# terminators). Keeping the terminator with the preceding chunk ("end") preserves the
# punctuation the model needs for prosody.
_SEPARATORS = [". ", "! ", "? ", "\n\n", "\n", " ", ""]


def chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split ``text`` into sentence-aligned chunks, each ``<= max_chars``.

    Short sentences are packed together so they share a chunk (and thus one connection);
    a break never lands mid-sentence unless a single sentence exceeds the budget, in which
    case that sentence alone is sliced. Whitespace-only input yields no chunks, and no
    text is dropped — rejoining the chunks recovers every word.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=0,
        separators=_SEPARATORS,
        keep_separator="end",
    )
    return splitter.split_text(text)
