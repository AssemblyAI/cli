"""Minimal local type stubs for the parts of jiwer that aai_cli.wer uses.

jiwer ships no type information, which would force `disallow_untyped_calls` off
(or a per-call `# type: ignore`). Typing just the surface we touch lets mypy run
the full --strict bar with no escape hatch, and documents the contract wer.py
relies on. Keep this in sync with aai_cli/wer.py — it is not the upstream API in
full, only what we call.
"""

from collections.abc import Sequence

class AbstractTransform:
    def __call__(self, sentences: str | list[str]) -> list[list[str]]: ...

class Compose(AbstractTransform):
    def __init__(self, transforms: Sequence[AbstractTransform]) -> None: ...

class ToLowerCase(AbstractTransform): ...
class RemovePunctuation(AbstractTransform): ...
class RemoveMultipleSpaces(AbstractTransform): ...
class Strip(AbstractTransform): ...
class ReduceToListOfListOfWords(AbstractTransform): ...

class WordOutput:
    hits: int
    substitutions: int
    deletions: int
    insertions: int

def process_words(
    reference: str | list[str],
    hypothesis: str | list[str],
    reference_transform: AbstractTransform = ...,
    hypothesis_transform: AbstractTransform = ...,
) -> WordOutput: ...
