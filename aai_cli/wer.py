"""Word error rate (WER) scoring for `assembly eval`, backed by jiwer.

A thin shim over :mod:`jiwer` — the de-facto standard WER implementation — so
the alignment math is never re-derived here. Texts are normalized the way ASR
benchmarks conventionally are (lowercase, punctuation stripped, whitespace
collapsed) so a transcript isn't penalized for casing or punctuation style.
jiwer is imported lazily inside the scoring functions (mirroring ``der.py``'s
lazy pyannote import) so this module stays import-cheap for the command layer —
an install that's missing the eval scoring stack must still run every other
command. No SDK, no Rich: the command layer owns all rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

if TYPE_CHECKING:
    import jiwer


def _transform() -> jiwer.Compose:
    """The normalization applied to both reference and hypothesis before alignment."""
    import jiwer

    return jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


# jiwer's transform output is only partially typed; validate the shape instead
# of trusting inference (the project pattern for untyped third-party returns).
_SENTENCES: TypeAdapter[list[list[str]]] = TypeAdapter(list[list[str]])


def normalize_words(text: str) -> list[str]:
    """The text as the list of comparison words jiwer will align.

    Exposed so the dataset loader can reject a reference that normalizes to
    nothing (e.g. punctuation-only) with the same rule the scorer uses.
    """
    return _SENTENCES.validate_python(_transform()(text))[0]


@dataclass(frozen=True)
class Score:
    """Edit errors against a reference of ``words`` words; pooled for corpus WER."""

    errors: int
    words: int

    @property
    def wer(self) -> float:
        return self.errors / self.words


def score(reference: str, hypothesis: str) -> Score:
    """Score a hypothesis transcript against its reference text.

    The caller guarantees a reference that normalizes to at least one word (the
    dataset loader rejects empty-reference rows), so ``Score.wer`` is always
    well-defined.
    """
    import jiwer

    transform = _transform()
    out = jiwer.process_words(
        reference,
        hypothesis,
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    return Score(
        errors=out.substitutions + out.deletions + out.insertions,
        words=out.hits + out.substitutions + out.deletions,
    )


def pooled(scores: list[Score]) -> Score:
    """Corpus-level score: total errors over total reference words (not a mean of rates)."""
    return Score(
        errors=sum(item.errors for item in scores),
        words=sum(item.words for item in scores),
    )
