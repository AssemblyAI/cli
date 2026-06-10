from __future__ import annotations

import re
from dataclasses import dataclass

# A rendered transcript line: "Speaker A: text". The id is a non-space run; the
# colon may be followed by a single space (label-only lines can lack trailing text).
_LABEL_RE = re.compile(r"^Speaker (?P<id>\S+): ?(?P<text>.*)$")


@dataclass(frozen=True)
class Segment:
    """One speaker turn after labels are stripped and wrapped lines folded in."""

    speaker_id: str
    text: str


def looks_like_speaker_labeled(text: str) -> bool:
    """True when the first non-blank line is a ``Speaker <id>:`` label line."""
    for line in text.splitlines():
        if line.strip():
            return _LABEL_RE.match(line) is not None
    return False


def parse_segments(text: str) -> list[Segment]:
    """Parse speaker-labeled text into merged per-turn segments.

    A ``Speaker <id>:`` line starts a turn; a following line that is not itself a
    label is a wrapped continuation appended to the current turn (joined with a
    single space). Consecutive same-speaker turns merge; empty turns are dropped.
    """
    turns: list[Segment] = []
    current_id: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if current_id is not None:
            turns.append(Segment(current_id, " ".join(p for p in parts if p)))

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _LABEL_RE.match(line)
        if match:
            flush()
            current_id = match.group("id")
            parts = [match.group("text").strip()]
        elif current_id is not None:
            parts.append(stripped)
    flush()

    merged: list[Segment] = []
    for turn in turns:
        if merged and merged[-1].speaker_id == turn.speaker_id:
            joined = f"{merged[-1].text} {turn.text}".strip()
            merged[-1] = Segment(turn.speaker_id, joined)
        else:
            merged.append(turn)
    return [turn for turn in merged if turn.text]
