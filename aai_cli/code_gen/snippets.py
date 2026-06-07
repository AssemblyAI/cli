from __future__ import annotations

from collections.abc import Callable

# (feature-name, enabled-predicate, result-handling code) in render order.
_Entry = tuple[str, Callable[[dict[str, object]], bool], str]


def _has(*keys: str) -> Callable[[dict[str, object]], bool]:
    return lambda merged: any(merged.get(k) for k in keys)


_SNIPPETS: list[_Entry] = [
    (
        "speaker_labels",
        _has("speaker_labels"),
        'for utt in transcript.utterances or []:\n    print(f"Speaker {utt.speaker}: {utt.text}")',
    ),
    (
        "summary",
        _has("summarization"),
        'if transcript.summary:\n    print("Summary:", transcript.summary)',
    ),
    (
        "chapters",
        _has("auto_chapters"),
        "for ch in transcript.chapters or []:\n    print(ch.headline)",
    ),
    (
        "highlights",
        _has("auto_highlights"),
        "results = getattr(transcript.auto_highlights, 'results', None) or []\n"
        'for h in results:\n    print(f"({h.count}x) {h.text}")',
    ),
    (
        "sentiment",
        _has("sentiment_analysis"),
        "for s in transcript.sentiment_analysis or []:\n    print(s.sentiment, s.text)",
    ),
    (
        "entities",
        _has("entity_detection"),
        'for ent in transcript.entities or []:\n    print(f"{ent.entity_type}: {ent.text}")',
    ),
    (
        "topics",
        _has("iab_categories"),
        "topic_summary = getattr(transcript.iab_categories, 'summary', None) or {}\n"
        "for label, relevance in topic_summary.items():\n    print(label, relevance)",
    ),
    (
        "content_safety",
        _has("content_safety"),
        "safety_summary = getattr(transcript.content_safety, 'summary', None) or {}\n"
        "for label, confidence in safety_summary.items():\n    print(label, confidence)",
    ),
]

# Feature names with a snippet — asserted complete by the coverage-guard test.
SNIPPET_FEATURES = [name for name, _pred, _code in _SNIPPETS]


def result_handling(merged: dict[str, object]) -> str:
    """Return result-handling code for the enabled analysis features.

    Falls back to printing the transcript text when no analysis feature is on.
    """
    blocks = [code for _name, pred, code in _SNIPPETS if pred(merged)]
    if not blocks:
        return "print(transcript.text)"
    return "\n\n".join(blocks)
