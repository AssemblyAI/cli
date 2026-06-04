from __future__ import annotations

from collections import Counter

from rich.console import Console
from rich.text import Text

from aai_cli import theme


def _fmt_ms(ms: int) -> str:
    total = int(ms) // 1000
    return f"{total // 60:02d}:{total % 60:02d}"


def _enum_value(obj: object) -> str:
    return str(getattr(obj, "value", obj))


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
    utterances = getattr(transcript, "utterances", None)
    if isinstance(utterances, list) and utterances:
        line = Text()
        for i, u in enumerate(utterances):
            if i:
                line.append("\n")
            line.append(f"Speaker {u.speaker}: ", style=theme.speaker_style(u.speaker))
            line.append(str(u.text))
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
    chapters = getattr(transcript, "chapters", None)
    if not chapters:
        return
    console.print("\n[bold]Chapters:[/bold]")
    for ch in chapters:
        span = f"{_fmt_ms(ch.start)}-{_fmt_ms(ch.end)}"
        console.print(f"  {span}  {ch.headline}")


def _render_highlights(transcript: object, console: Console) -> None:
    highlights = getattr(transcript, "auto_highlights", None)
    results = getattr(highlights, "results", None) if highlights else None
    if not results:
        return
    console.print("\n[bold]Highlights:[/bold]")
    for h in results:
        console.print(f"  ({h.count}x) {h.text}")


def _render_sentiment(transcript: object, console: Console) -> None:
    results = getattr(transcript, "sentiment_analysis", None)
    if not results:
        return
    counts = Counter(_enum_value(r.sentiment).lower() for r in results)
    total = sum(counts.values()) or 1
    parts = [f"{pct * 100 // total}% {label}" for label, pct in counts.items()]
    console.print("\n[bold]Sentiment:[/bold] " + ", ".join(parts))


def _render_entities(transcript: object, console: Console) -> None:
    entities = getattr(transcript, "entities", None)
    if not entities:
        return
    console.print("\n[bold]Entities:[/bold]")
    for ent in entities:
        console.print(f"  {_enum_value(ent.entity_type)}: {ent.text}")


def _render_topics(transcript: object, console: Console) -> None:
    iab = getattr(transcript, "iab_categories", None)
    summary = getattr(iab, "summary", None) if iab else None
    if not summary:
        return
    console.print("\n[bold]Topics:[/bold]")
    for label, relevance in sorted(summary.items(), key=lambda kv: kv[1], reverse=True):
        console.print(f"  {label} ({float(relevance):.2f})")


def _render_content_safety(transcript: object, console: Console) -> None:
    safety = getattr(transcript, "content_safety", None)
    summary = getattr(safety, "summary", None) if safety else None
    if not summary:
        return
    console.print("\n[bold]Content Safety:[/bold]")
    for label, confidence in sorted(summary.items(), key=lambda kv: kv[1], reverse=True):
        console.print(f"  {_enum_value(label)} ({float(confidence):.2f})")
