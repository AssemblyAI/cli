"""Segment selection for `assembly clip`: pure logic, no I/O.

Everything here turns user selectors into :class:`Segment` lists — parsing
``--range`` values, filtering diarized utterances for ``--speaker``/``--search``,
rendering the timestamped listing an ``--llm`` model selects from, parsing the
model's reply, merging the combined selection, and snapping the merged
boundaries into detected silence. The orchestration (transcript fetch, LLM
call, ffmpeg — including the silencedetect pass) lives in ``clip_exec``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from aai_cli import jsonshape
from aai_cli.errors import CLIError, UsageError

_RANGE_FORMAT = "START-END, each end as seconds or [HH:]MM:SS (e.g. 90-120 or 1:30-2:00)"
_MAX_CLOCK_FIELDS = 3  # [HH:]MM:SS — anything longer than three colon fields is a typo


@dataclass(frozen=True)
class Segment:
    """A time window within the source media, in seconds."""

    start: float
    end: float


def _bad_range(flag_value: str) -> UsageError:
    return UsageError(
        f"Invalid --range {flag_value!r}.",
        suggestion=f"Use {_RANGE_FORMAT}.",
    )


def _parse_point(token: str, flag_value: str) -> float:
    """Seconds for one ``--range`` endpoint: bare seconds or colon-separated clock time."""
    parts = token.strip().split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        raise _bad_range(flag_value) from None
    if len(values) > _MAX_CLOCK_FIELDS or any(not math.isfinite(value) for value in values):
        raise _bad_range(flag_value)
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def parse_range(flag_value: str) -> Segment:
    """The :class:`Segment` for one ``--range START-END`` flag value.

    Negative endpoints can't be expressed (``-`` is the separator), so the only
    validations are shape, finiteness, and end-after-start.
    """
    head, sep, tail = flag_value.partition("-")
    if not sep:
        raise _bad_range(flag_value)
    segment = Segment(_parse_point(head, flag_value), _parse_point(tail, flag_value))
    if segment.end <= segment.start:
        raise UsageError(
            f"--range end must be after its start: {flag_value!r}.",
            suggestion=f"Use {_RANGE_FORMAT}.",
        )
    return segment


def merge_segments(segments: list[Segment], padding: float) -> list[Segment]:
    """Padded segments, sorted and coalesced where they touch or overlap.

    Padding widens each segment on both sides (clamped at 0); overlapping or
    back-to-back selections fold into one clip so a speaker's consecutive
    utterances don't shatter into per-sentence files.
    """
    padded = sorted(
        (Segment(max(0.0, seg.start - padding), seg.end + padding) for seg in segments),
        key=lambda seg: seg.start,
    )
    merged: list[Segment] = []
    for seg in padded:
        if merged and seg.start <= merged[-1].end:
            merged[-1] = Segment(merged[-1].start, max(merged[-1].end, seg.end))
        else:
            merged.append(seg)
    return merged


# One silencedetect log edge: "silence_start: 12.34" or "silence_end: 13.01".
_SILENCE_EDGE = re.compile(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)")

SNAP_REACH = 1.5  # how far (seconds) a boundary may move to reach silence
SNAP_LEAD = 0.25  # silence kept next to the speech once a boundary snaps


def parse_silences(detect_log: str) -> list[Segment]:
    """The silence intervals in ffmpeg ``silencedetect`` output, in order.

    A trailing ``silence_start`` with no matching end runs to end-of-file; a
    small negative start (decoder priming samples) clamps to 0.
    """
    silences: list[Segment] = []
    pending: float | None = None
    for kind, value in _SILENCE_EDGE.findall(detect_log):
        if kind == "start":
            pending = max(0.0, float(value))
        elif pending is not None:
            silences.append(Segment(pending, float(value)))
            pending = None
    if pending is not None:
        silences.append(Segment(pending, math.inf))
    return silences


def _snap_start(t: float, silences: list[Segment]) -> float:
    """A clip start moved back into the silence just before its speech.

    A start already inside silence stays put (it honors --padding exactly);
    one that lands mid-speech moves into the last silence before it, to
    ``SNAP_LEAD`` before the speech resumes — only ever widening the clip, so
    selected content is never dropped. With no silence within ``SNAP_REACH``
    (continuous speech) it stays where asked. ``silences`` must be sorted.
    """
    gap = next((s for s in reversed(silences) if s.start <= t), None)
    if gap is None:
        return t  # before the first silence ever begins
    if t <= gap.end:
        return t  # already in silence
    if t - gap.end > SNAP_REACH:
        return t  # continuous speech as far back as snapping may reach
    return max(gap.start, gap.end - SNAP_LEAD)


def _snap_end(t: float, silences: list[Segment]) -> float:
    """A clip end moved forward into the next silence (mirror of _snap_start)."""
    gap = next((s for s in silences if s.end >= t), None)
    if gap is None:
        return t  # after the last silence ends
    if gap.start <= t:
        return t  # already in silence
    if gap.start - t > SNAP_REACH:
        return t
    return min(gap.end, gap.start + SNAP_LEAD)


def snap_to_silences(segments: list[Segment], silences: list[Segment]) -> list[Segment]:
    """Segments with any boundary that lands on speech moved into adjacent
    silence, so cuts don't fall mid-word; re-coalesced afterwards since a
    start moving back can reach the previous segment's (moved) end."""
    if not silences:
        return segments
    ordered = sorted(silences, key=lambda s: s.start)
    snapped = [
        Segment(_snap_start(seg.start, ordered), _snap_end(seg.end, ordered)) for seg in segments
    ]
    return merge_segments(snapped, 0.0)


def matching_utterances(
    utterances: list[object], speakers: list[str], search: str | None
) -> list[object]:
    """The utterances passing the ``--speaker``/``--search`` filters.

    Both filters are case-insensitive and combine with AND; an unset filter
    passes everything.
    """
    wanted = {speaker.upper() for speaker in speakers}
    needle = search.lower() if search is not None else None
    matched: list[object] = []
    for utterance in utterances:
        speaker = str(getattr(utterance, "speaker", "") or "")
        text = str(getattr(utterance, "text", "") or "")
        if wanted and speaker.upper() not in wanted:
            continue
        if needle is not None and needle not in text.lower():
            continue
        matched.append(utterance)
    return matched


def segment_of(utterance: object) -> Segment:
    """The utterance's time window in seconds (the API reports milliseconds)."""
    start_ms = jsonshape.as_float(getattr(utterance, "start", None))
    end_ms = jsonshape.as_float(getattr(utterance, "end", None))
    return Segment(start_ms / 1000.0, end_ms / 1000.0)


def utterance_listing(utterances: list[object]) -> str:
    """The timestamped transcript view the LLM selects from, one utterance per line."""
    lines: list[str] = []
    for utterance in utterances:
        seg = segment_of(utterance)
        speaker = str(getattr(utterance, "speaker", "") or "")
        text = str(getattr(utterance, "text", "") or "")
        lines.append(f"[{seg.start:.3f}-{seg.end:.3f}] {speaker}: {text}")
    return "\n".join(lines)


# Prefixed to the user's --llm instruction; the reply contract ("only a JSON
# array") is what parse_llm_segments depends on.
LLM_INSTRUCTIONS = (
    "Select the time ranges to cut from the timestamped transcript below. "
    'Reply with only a JSON array like [{"start": 12.5, "end": 30.0}] — '
    "start/end in seconds within the transcript, no prose, no code fences. "
    "Selection instruction: "
)


def _llm_range_items(reply: str) -> list[dict[str, object]] | None:
    """The JSON array of range objects in the model's reply, or None.

    Tolerates prose or code fences around the array by slicing from the first
    ``[`` to the last ``]``; anything that doesn't decode to a list of objects
    is a parse failure.
    """
    try:
        loaded: object = json.loads(reply[reply.find("[") : reply.rfind("]") + 1])
    except json.JSONDecodeError:
        return None
    return jsonshape.as_object_list(loaded)


def _segment_from_item(item: dict[str, object]) -> Segment | None:
    """A Segment from one model-returned range object, or None for a malformed
    entry (wrong types, non-finite, negative, or inverted) — one bad entry is
    dropped rather than failing the whole selection."""
    start, end = item.get("start"), item.get("end")
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        return None
    segment = Segment(float(start), float(end))
    if not math.isfinite(segment.start) or not math.isfinite(segment.end):
        return None
    if segment.start < 0 or segment.end <= segment.start:
        return None
    return segment


def parse_llm_segments(reply: str) -> list[Segment]:
    """The segments the model selected, parsed defensively from its reply."""
    items = _llm_range_items(reply)
    if items is None:
        raise CLIError(
            "The model's reply could not be read as clip ranges.",
            error_type="llm_parse_error",
            suggestion=(
                "Re-run, or rephrase --llm; the model must answer with a JSON array "
                'of {"start", "end"} seconds.'
            ),
        )
    segments = [seg for item in items if (seg := _segment_from_item(item)) is not None]
    if not segments:
        raise CLIError(
            "The model selected no segments.",
            error_type="no_match",
            suggestion="Loosen the --llm instruction, or select with --speaker/--search/--range.",
        )
    return segments


def format_clock(seconds: float) -> str:
    """``M:SS.t`` (or ``H:MM:SS.t``) for the human view of a clip window."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:04.1f}"
    return f"{minutes}:{secs:04.1f}"
