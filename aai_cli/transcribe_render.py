from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from rich.console import Console
from rich.text import Text

from aai_cli import jsonshape, theme


def _fmt_ms(ms: int) -> str:
    total = int(ms) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _enum_value(obj: object) -> str:
    return str(getattr(obj, "value", obj))


def _nested(transcript: object, outer: str, inner: str) -> object | None:
    """Return ``transcript.<outer>.<inner>``, or None if either link is missing/falsy."""
    mid = getattr(transcript, outer, None)
    return getattr(mid, inner, None) if mid else None


def _objects(value: object) -> list[object]:
    return jsonshape.object_list(value)


def _mapping(value: object) -> dict[str, object]:
    return jsonshape.as_mapping(value) or {}


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def render_transcript_result(transcript: object, console: Console) -> None:
    """Print the transcript text, then a section per analysis feature present."""
    _render_text(transcript, console)
    _render_summary(transcript, console)
    _render_chapters(transcript, console)
    _render_highlights(transcript, console)
    _render_sentiment(transcript, console)
    _render_entities(transcript, console)
    _render_topics(transcript, console)
    _render_content_safety(transcript, console)


def _render_text(transcript: object, console: Console) -> None:
    """Print per-speaker utterances when present, else the flat transcript text."""
    utterances = _objects(getattr(transcript, "utterances", None))
    if utterances:
        line = Text()
        for i, u in enumerate(utterances):
            if i:
                line.append("\n")
            speaker = getattr(u, "speaker", "")
            line.append(f"Speaker {speaker}: ", style=theme.speaker_style(speaker))
            line.append(str(getattr(u, "text", "")))
        console.print(line)
        return
    # Wrap in Text so transcript content with [brackets] is not parsed as Rich markup.
    console.print(Text(getattr(transcript, "text", "") or ""))


def _render_summary(transcript: object, console: Console) -> None:
    summary = getattr(transcript, "summary", None)
    if summary:
        console.print("\n[bold]Summary:[/bold]")
        console.print(str(summary))


def _render_chapters(transcript: object, console: Console) -> None:
    chapters = _objects(getattr(transcript, "chapters", None))
    if not chapters:
        return
    console.print("\n[bold]Chapters:[/bold]")
    for ch in chapters:
        span = f"{_fmt_ms(_int_value(getattr(ch, 'start', 0)))}-{_fmt_ms(_int_value(getattr(ch, 'end', 0)))}"
        console.print(f"  {span}  {getattr(ch, 'headline', '')}")


def _render_highlights(transcript: object, console: Console) -> None:
    results = _objects(_nested(transcript, "auto_highlights", "results"))
    if not results:
        return
    console.print("\n[bold]Highlights:[/bold]")
    for h in results:
        console.print(f"  ({getattr(h, 'count', '')}x) {getattr(h, 'text', '')}")


def _render_sentiment(transcript: object, console: Console) -> None:
    results = _objects(getattr(transcript, "sentiment_analysis", None))
    if not results:
        return
    counts = Counter(_enum_value(getattr(r, "sentiment", "")).lower() for r in results)
    total = sum(counts.values()) or 1
    parts = [f"{pct * 100 // total}% {label}" for label, pct in counts.items()]
    console.print("\n[bold]Sentiment:[/bold] " + ", ".join(parts))


def _render_entities(transcript: object, console: Console) -> None:
    entities = _objects(getattr(transcript, "entities", None))
    if not entities:
        return
    console.print("\n[bold]Entities:[/bold]")
    for ent in entities:
        console.print(
            f"  {_enum_value(getattr(ent, 'entity_type', ''))}: {getattr(ent, 'text', '')}"
        )


def _render_topics(transcript: object, console: Console) -> None:
    summary = _mapping(_nested(transcript, "iab_categories", "summary"))
    if not summary:
        return
    console.print("\n[bold]Topics:[/bold]")
    items: Mapping[str, object] = summary
    for label, relevance in sorted(items.items(), key=lambda kv: _float_value(kv[1]), reverse=True):
        console.print(f"  {label} ({_float_value(relevance):.2f})")


def _render_content_safety(transcript: object, console: Console) -> None:
    summary = _mapping(_nested(transcript, "content_safety", "summary"))
    if not summary:
        return
    console.print("\n[bold]Content Safety:[/bold]")
    items: Mapping[str, object] = summary
    for label, confidence in sorted(
        items.items(), key=lambda kv: _float_value(kv[1]), reverse=True
    ):
        console.print(f"  {_enum_value(label)} ({_float_value(confidence):.2f})")
