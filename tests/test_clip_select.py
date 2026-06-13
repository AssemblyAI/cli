"""Tests for the pure clip selection logic (aai_cli/clip_select.py): --range
parsing, segment merging, utterance filtering, the LLM listing/reply contract,
silencedetect parsing, boundary snapping, and clock formatting."""

from __future__ import annotations

import math

import pytest

from aai_cli.commands.clip import _select as clip_select
from aai_cli.commands.clip._select import Segment
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


# --- silence detection & snapping ---------------------------------------------


def test_parse_silences_pairs_start_end_edges_in_order():
    log = (
        "Stream mapping: ...\n"
        "[silencedetect @ 0x6000] silence_start: 4\n"
        "[silencedetect @ 0x6000] silence_end: 4.6 | silence_duration: 0.6\n"
        "[silencedetect @ 0x6000] silence_start: 13.25\n"
        "[silencedetect @ 0x6000] silence_end: 14 | silence_duration: 0.75\n"
    )
    assert clip_select.parse_silences(log) == [Segment(4.0, 4.6), Segment(13.25, 14.0)]


def test_parse_silences_trailing_start_runs_to_end_of_file():
    assert clip_select.parse_silences("silence_start: 7.5\n") == [Segment(7.5, math.inf)]


def test_parse_silences_clamps_negative_start_to_zero():
    # ffmpeg can report a small negative start from decoder priming samples.
    log = "silence_start: -0.011\nsilence_end: 1.5 | silence_duration: 1.511\n"
    assert clip_select.parse_silences(log) == [Segment(0.0, 1.5)]


def test_parse_silences_ignores_an_unpaired_end():
    log = "silence_start: 1\nsilence_end: 2\nsilence_end: 3\n"
    assert clip_select.parse_silences(log) == [Segment(1.0, 2.0)]


def test_parse_silences_empty_log_finds_nothing():
    assert clip_select.parse_silences("") == []


SILENCES = [Segment(4.0, 4.6), Segment(13.0, 14.0)]


def test_snap_moves_speech_boundaries_into_adjacent_silence():
    # 5.0 and 12.5 both land on speech: the start backs into the 4.0-4.6
    # silence (SNAP_LEAD before speech resumes at 4.6), the end runs forward
    # into the 13.0-14.0 silence (SNAP_LEAD past where speech stops at 13.0).
    snapped = clip_select.snap_to_silences([Segment(5.0, 12.5)], SILENCES)
    assert snapped == [Segment(4.35, 13.25)]


def test_snap_clamps_inside_a_narrow_silence():
    # Silences narrower than SNAP_LEAD: the boundary stays within the silence.
    silences = [Segment(4.5, 4.6), Segment(13.0, 13.1)]
    snapped = clip_select.snap_to_silences([Segment(5.0, 12.5)], silences)
    assert snapped == [Segment(4.5, 13.1)]


def test_snap_leaves_boundaries_already_in_silence_alone():
    # Both boundaries sit in silence already (one mid-silence, one exactly at
    # a silence edge): they honor the selection (and --padding) exactly.
    snapped = clip_select.snap_to_silences([Segment(4.2, 13.0)], SILENCES)
    assert snapped == [Segment(4.2, 13.0)]
    snapped = clip_select.snap_to_silences([Segment(4.6, 14.0)], SILENCES)
    assert snapped == [Segment(4.6, 14.0)]


def test_snap_prefers_the_silence_a_boundary_touches_over_a_nearby_one():
    # A start exactly at a silence's start (and an end exactly at a silence's
    # end) belongs to that silence — not snapped toward the neighbouring one.
    silences = [Segment(3.2, 4.4), Segment(4.6, 5.0), Segment(5.2, 6.0)]
    snapped = clip_select.snap_to_silences([Segment(4.6, 5.0)], silences)
    assert snapped == [Segment(4.6, 5.0)]


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        # Exactly SNAP_REACH from the silences on both sides: still snaps.
        (Segment(5.0, 12.5), Segment(3.25, 14.25)),
        # Just beyond SNAP_REACH on both sides: continuous speech, stays put.
        (Segment(5.1, 12.4), Segment(5.1, 12.4)),
    ],
)
def test_snap_reach_bounds_how_far_a_boundary_moves(segment, expected):
    silences = [Segment(1.0, 3.5), Segment(14.0, 15.0)]
    assert clip_select.snap_to_silences([segment], silences) == [expected]


def test_snap_boundaries_beyond_all_silences_stay_put():
    snapped = clip_select.snap_to_silences([Segment(0.5, 20.0)], SILENCES)
    assert snapped == [Segment(0.5, 20.0)]


def test_snap_end_into_a_trailing_silence_that_runs_to_end_of_file():
    silences = [Segment(10.0, math.inf)]
    snapped = clip_select.snap_to_silences([Segment(5.0, 9.9)], silences)
    assert snapped == [Segment(5.0, 10.25)]


def test_snap_sorts_the_silences_it_is_given():
    silences = [Segment(13.0, 14.0), Segment(4.0, 4.6)]
    snapped = clip_select.snap_to_silences([Segment(14.2, 20.0)], silences)
    assert snapped == [Segment(13.75, 20.0)]


def test_snap_merges_segments_that_meet_inside_one_silence():
    # Both boundaries snap into the same 4.6-5.0 silence and now overlap, so
    # the two clips fold into one instead of duplicating the pause.
    silences = [Segment(4.6, 5.0)]
    snapped = clip_select.snap_to_silences([Segment(2.0, 4.5), Segment(5.2, 7.0)], silences)
    assert snapped == [Segment(2.0, 7.0)]


def test_snap_without_silences_changes_nothing():
    # No detected silences (or a failed detection): segments pass through
    # untouched — not even re-merged.
    segments = [Segment(5.0, 6.0), Segment(5.5, 7.0)]
    assert clip_select.snap_to_silences(segments, []) == segments


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
