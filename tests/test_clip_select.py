"""Tests for the pure clip selection logic (aai_cli/clip_select.py): --range
parsing, segment merging, utterance filtering, the LLM listing/reply contract,
and clock formatting."""

from __future__ import annotations

import pytest

from aai_cli import clip_select
from aai_cli.clip_select import Segment
from aai_cli.errors import CLIError, UsageError
from tests._clip_helpers import UTTERANCES

# --- range parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    ("flag_value", "start", "end"),
    [
        ("5-12.5", 5.0, 12.5),
        ("90-120", 90.0, 120.0),
        ("1:30-2:45", 90.0, 165.0),
        ("1:00:03.5-1:00:04", 3603.5, 3604.0),
        ("5 - 10", 5.0, 10.0),
        ("0-0.5", 0.0, 0.5),
    ],
)
def test_parse_range_accepts_seconds_and_clock_times(flag_value, start, end):
    assert clip_select.parse_range(flag_value) == Segment(start, end)


@pytest.mark.parametrize(
    "flag_value",
    ["5", "5-", "-5", "abc-10", "5-10-15", "1:2:3:4-5", "inf-10", "nan-10", "1e400-2e400"],
)
def test_parse_range_rejects_malformed_values(flag_value):
    with pytest.raises(UsageError) as exc:
        clip_select.parse_range(flag_value)
    # Specifically the malformed-shape error — "1:2:3:4" must not parse as a
    # huge clock value and fall through to the end-before-start error instead.
    assert "Invalid --range" in exc.value.message
    assert flag_value in exc.value.message
    assert "START-END" in (exc.value.suggestion or "")


@pytest.mark.parametrize("flag_value", ["10-5", "5-5"])
def test_parse_range_rejects_end_not_after_start(flag_value):
    with pytest.raises(UsageError) as exc:
        clip_select.parse_range(flag_value)
    assert "end must be after its start" in exc.value.message


# --- segment merging ---------------------------------------------------------


def test_merge_segments_sorts_disjoint_segments():
    segs = [Segment(10.0, 11.0), Segment(0.0, 1.0)]
    assert clip_select.merge_segments(segs, 0.0) == [Segment(0.0, 1.0), Segment(10.0, 11.0)]


def test_merge_segments_coalesces_overlapping_and_touching():
    assert clip_select.merge_segments([Segment(0.0, 5.0), Segment(4.0, 8.0)], 0.0) == [
        Segment(0.0, 8.0)
    ]
    # Back-to-back (start == previous end) folds too — `<=`, not `<`.
    assert clip_select.merge_segments([Segment(0.0, 5.0), Segment(5.0, 8.0)], 0.0) == [
        Segment(0.0, 8.0)
    ]


def test_merge_segments_keeps_outer_end_for_contained_segment():
    assert clip_select.merge_segments([Segment(0.0, 10.0), Segment(2.0, 3.0)], 0.0) == [
        Segment(0.0, 10.0)
    ]


def test_merge_segments_merges_against_the_last_segment_only():
    segs = [Segment(0.0, 1.0), Segment(5.0, 6.0), Segment(10.0, 11.0), Segment(10.5, 12.0)]
    assert clip_select.merge_segments(segs, 0.0) == [
        Segment(0.0, 1.0),
        Segment(5.0, 6.0),
        Segment(10.0, 12.0),
    ]


def test_merge_segments_padding_widens_and_clamps_at_zero():
    assert clip_select.merge_segments([Segment(0.2, 1.0)], 0.5) == [Segment(0.0, 1.5)]


def test_merge_segments_padding_bridges_a_small_gap():
    merged = clip_select.merge_segments([Segment(0.0, 1.0), Segment(1.5, 2.0)], 0.3)
    assert merged == [Segment(0.0, 2.0 + 0.3)]


# --- utterance selection -----------------------------------------------------


def _filtered_segments(utterances, speakers, search):
    matched = clip_select.matching_utterances(utterances, speakers, search)
    return [clip_select.segment_of(u) for u in matched]


def test_utterance_segments_converts_milliseconds_to_seconds():
    segs = _filtered_segments(list(UTTERANCES), [], "sounds")
    assert segs == [Segment(3.0, 4.0)]


def test_utterance_segments_speaker_filter_is_case_insensitive():
    segs = _filtered_segments(list(UTTERANCES), ["a"], None)
    assert segs == [Segment(1.5, 2.5), Segment(5.0, 6.0)]


def test_utterance_segments_search_is_case_insensitive():
    segs = _filtered_segments(list(UTTERANCES), [], "PRICING")
    assert segs == [Segment(1.5, 2.5)]


def test_utterance_segments_speaker_and_search_combine_with_and():
    segs = _filtered_segments(list(UTTERANCES), ["A"], "hiring")
    assert segs == [Segment(5.0, 6.0)]


def test_utterance_segments_excludes_unselected_speakers():
    segs = _filtered_segments(list(UTTERANCES), ["B"], None)
    assert segs == [Segment(3.0, 4.0)]


# --- the LLM listing / reply contract ------------------------------------------


def test_utterance_listing_renders_timestamped_lines():
    listing = clip_select.utterance_listing(list(UTTERANCES))
    assert listing == (
        "[1.500-2.500] A: Let's talk pricing today.\n"
        "[3.000-4.000] B: Sounds good.\n"
        "[5.000-6.000] A: Moving on to hiring."
    )


@pytest.mark.parametrize(
    "reply",
    [
        '[{"start": 5, "end": 9.5}]',
        '```json\n[{"start": 5, "end": 9.5}]\n```',
        'Here are the ranges: [{"start": 5, "end": 9.5}] - enjoy!',
        # The slice must stop exactly at the closing "]" — the next char would
        # break the JSON.
        '[{"start": 5, "end": 9.5}], thanks',
    ],
)
def test_parse_llm_segments_reads_the_array_through_noise(reply):
    assert clip_select.parse_llm_segments(reply) == [Segment(5.0, 9.5)]


@pytest.mark.parametrize("reply", ["no ranges here", "", "[1, 2, 3]", "[{]"])
def test_parse_llm_segments_rejects_unreadable_replies(reply):
    with pytest.raises(CLIError) as exc:
        clip_select.parse_llm_segments(reply)
    assert exc.value.error_type == "llm_parse_error"
    assert "could not be read as clip ranges" in exc.value.message
    assert "JSON array" in (exc.value.suggestion or "")


def test_parse_llm_segments_errors_when_model_selects_nothing():
    with pytest.raises(CLIError) as exc:
        clip_select.parse_llm_segments("[]")
    assert exc.value.error_type == "no_match"
    assert "The model selected no segments" in exc.value.message
    assert "--speaker/--search/--range" in (exc.value.suggestion or "")


def test_parse_llm_segments_drops_malformed_entries():
    reply = (
        '[{"start": 0.5, "end": 0.9},'
        ' {"start": "x", "end": 2},'
        ' {"start": 3},'
        ' {"start": -1, "end": 2},'
        ' {"start": 4, "end": 4},'
        ' {"start": 9, "end": 5},'
        ' {"start": Infinity, "end": 10},'
        ' {"start": 1, "end": Infinity},'
        ' {"start": 6, "end": 7.5}]'
    )
    assert clip_select.parse_llm_segments(reply) == [Segment(0.5, 0.9), Segment(6.0, 7.5)]


def test_segment_is_immutable():
    import dataclasses

    segment = Segment(0.0, 1.0)
    field_name = dataclasses.fields(segment)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(segment, field_name, 5.0)


# --- clock formatting --------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [
        (5.0, "0:05.0"),
        (90.5, "1:30.5"),
        (3723.5, "1:02:03.5"),
        (0.0, "0:00.0"),
    ],
)
def test_format_clock(seconds, rendered):
    assert clip_select.format_clock(seconds) == rendered
