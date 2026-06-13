from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from aai_cli.core.errors import UsageError

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
            joined = f"{merged[-1].text} {turn.text}"
            merged[-1] = Segment(turn.speaker_id, joined)
        else:
            merged.append(turn)
    return [turn for turn in merged if turn.text]


def parse_voice_overrides(values: list[str]) -> tuple[str | None, dict[str, str]]:
    """Split repeatable ``--voice`` values into ``(bare_voice, {speaker_id: voice})``.

    A value containing ``=`` is a ``SPEAKER=VOICE`` mapping (ids casefolded so the
    match is case-insensitive); a bare value sets the single-voice default, last
    one wins. Raises ``UsageError`` on a malformed pair (empty side).
    """
    bare: str | None = None
    overrides: dict[str, str] = {}
    for value in values:
        if "=" in value:
            speaker, _, voice = value.partition("=")
            speaker, voice = speaker.strip(), voice.strip()
            if not speaker or not voice:
                raise UsageError(
                    f"Invalid --voice mapping {value!r}.",
                    suggestion="Use SPEAKER=VOICE, e.g. --voice A=jane.",
                )
            overrides[speaker.casefold()] = voice
        elif value.strip():
            bare = value.strip()
    return bare, overrides


def assign_voices(
    segments: list[Segment],
    rotation: Sequence[str],
    overrides: dict[str, str],
) -> tuple[list[tuple[str, str]], dict[str, str]]:
    """Resolve each segment to ``(voice, text)`` and return the id→voice map.

    A speaker uses its override if present, else the next rotation voice in
    first-appearance order (wrapping). Overrides do not consume rotation slots.
    The returned map is ordered by first appearance.
    """
    id_to_voice: dict[str, str] = {}
    next_index = 0
    resolved: list[tuple[str, str]] = []
    for segment in segments:
        if segment.speaker_id not in id_to_voice:
            override = overrides.get(segment.speaker_id.casefold())
            if override is not None:
                id_to_voice[segment.speaker_id] = override
            else:
                id_to_voice[segment.speaker_id] = rotation[next_index % len(rotation)]
                next_index += 1
        resolved.append((id_to_voice[segment.speaker_id], segment.text))
    return resolved, id_to_voice
