from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from rich.console import Console
from rich.text import Text

from aai_cli import jsonshape, theme


def _fmt_ms(ms: int) -> str:
    """``MM:SS`` for sub-hour timestamps, ``H:MM:SS`` past that (so 1h isn't '60:00')."""
    total = int(ms) // 1000
    hours, rem = divmod(total, 3600)
    if hours:
        return f"{hours}:{rem // 60:02d}:{rem % 60:02d}"
    return f"{rem // 60:02d}:{rem % 60:02d}"


def _enum_value(obj: object) -> str:
    return str(getattr(obj, "value", obj))


def _nested(transcript: object, outer: str, inner: str) -> object | None:
    """Return ``transcript.<outer>.<inner>``, or None if either link is missing/falsy."""
    mid = getattr(transcript, outer, None)
    return getattr(mid, inner, None) if mid else None


def _mapping(value: object) -> dict[str, object]:
    return jsonshape.as_mapping(value) or {}


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
    utterances = jsonshape.object_list(getattr(transcript, "utterances", None))
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
    chapters = jsonshape.object_list(getattr(transcript, "chapters", None))
    if not chapters:
        return
    console.print("\n[bold]Chapters:[/bold]")
    for ch in chapters:
        # The `, 0` getattr fallbacks are equivalent mutants: they apply only to a
        # chapter missing start/end, and _fmt_ms(0) == _fmt_ms(1) == "00:00" regardless.
        span = f"{_fmt_ms(jsonshape.as_int(getattr(ch, 'start', 0)))}-{_fmt_ms(jsonshape.as_int(getattr(ch, 'end', 0)))}"  # pragma: no mutate
        console.print(f"  {span}  {getattr(ch, 'headline', '')}")


def _render_highlights(transcript: object, console: Console) -> None:
    results = jsonshape.object_list(_nested(transcript, "auto_highlights", "results"))
    if not results:
        return
    console.print("\n[bold]Highlights:[/bold]")
    for h in results:
        console.print(f"  ({getattr(h, 'count', '')}x) {getattr(h, 'text', '')}")


def _render_sentiment(transcript: object, console: Console) -> None:
    results = jsonshape.object_list(getattr(transcript, "sentiment_analysis", None))
    if not results:
        return
    counts = Counter(_enum_value(getattr(r, "sentiment", "")).lower() for r in results)
    parts = [f"{pct}% {label}" for label, pct in _percentages(counts).items()]
    console.print("\n[bold]Sentiment:[/bold] " + ", ".join(parts))


def _percentages(counts: Counter[str]) -> dict[str, int]:
    """Integer percentages that sum to exactly 100, in ``counts`` order.

    Plain flooring loses the rounding remainder (three equal counts -> 33+33+33 = 99),
    so the leftover points go to the labels with the largest fractional parts.
    """
    total = sum(counts.values())
    if not total:
        return {}
    shares = {label: count * 100 / total for label, count in counts.items()}
    pcts = {label: int(share) for label, share in shares.items()}
    leftover = 100 - sum(pcts.values())
    by_fraction = sorted(shares, key=lambda label: shares[label] - pcts[label], reverse=True)
    for label in by_fraction[:leftover]:
        pcts[label] += 1
    return pcts


def _render_entities(transcript: object, console: Console) -> None:
    entities = jsonshape.object_list(getattr(transcript, "entities", None))
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
    for label, relevance in sorted(
        items.items(), key=lambda kv: jsonshape.as_float(kv[1]), reverse=True
    ):
        console.print(f"  {label} ({jsonshape.as_float(relevance):.2f})")


def _render_content_safety(transcript: object, console: Console) -> None:
    summary = _mapping(_nested(transcript, "content_safety", "summary"))
    if not summary:
        return
    console.print("\n[bold]Content Safety:[/bold]")
    items: Mapping[str, object] = summary
    for label, confidence in sorted(
        items.items(), key=lambda kv: jsonshape.as_float(kv[1]), reverse=True
    ):
        console.print(f"  {_enum_value(label)} ({jsonshape.as_float(confidence):.2f})")
