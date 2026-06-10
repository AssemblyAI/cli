from __future__ import annotations

import pytest

from aai_cli.errors import UsageError
from aai_cli.tts import dialogue


def test_detects_labeled_input_on_first_nonblank_line():
    assert dialogue.looks_like_speaker_labeled("Speaker A: hi\nSpeaker B: yo") is True
    # Leading blank lines are skipped before the check.
    assert dialogue.looks_like_speaker_labeled("\n\nSpeaker A: hi") is True


def test_plain_prose_is_not_labeled():
    assert dialogue.looks_like_speaker_labeled("Hello there, friend.") is False
    # A mid-sentence "Speaker" word must not trigger detection.
    assert dialogue.looks_like_speaker_labeled("The Speaker said hello") is False
    assert dialogue.looks_like_speaker_labeled("") is False


def test_parse_single_line_turns():
    segs = dialogue.parse_segments("Speaker A: Hello.\nSpeaker B: Hi there.")
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "Hello."), ("B", "Hi there.")]


def test_parse_folds_wrapped_continuation_lines():
    # An utterance wrapped across physical lines (no label on the 2nd line) folds
    # back into one segment, joined with single spaces.
    text = "Speaker A: This is a long line that wrapped\nonto a second line here\nSpeaker B: Ok."
    segs = dialogue.parse_segments(text)
    assert [(s.speaker_id, s.text) for s in segs] == [
        ("A", "This is a long line that wrapped onto a second line here"),
        ("B", "Ok."),
    ]


def test_parse_merges_consecutive_same_speaker_turns():
    segs = dialogue.parse_segments("Speaker A: One.\nSpeaker A: Two.\nSpeaker B: Three.")
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "One. Two."), ("B", "Three.")]


def test_parse_skips_blank_lines_and_drops_empty_turns():
    segs = dialogue.parse_segments("Speaker A: Hi.\n\nSpeaker B: \nSpeaker A: Bye.")
    # Speaker B's empty turn is dropped; the two A turns are not merged (B is between).
    assert [(s.speaker_id, s.text) for s in segs] == [("A", "Hi."), ("A", "Bye.")]


def test_parse_label_only_line_with_continuation():
    # A label line with no inline text, followed by a wrapped continuation line:
    # the empty first part must be filtered so the text has no leading space.
    segs = dialogue.parse_segments("Speaker A:\ncontinuation text\nSpeaker B: Ok.")
    assert [(s.speaker_id, s.text) for s in segs] == [
        ("A", "continuation text"),
        ("B", "Ok."),
    ]


def test_parse_voice_overrides_splits_bare_and_mapped():
    bare, overrides = dialogue.parse_voice_overrides(["A=vera", "mary", "B=paul"])
    assert bare == "mary"
    assert overrides == {"a": "vera", "b": "paul"}  # ids casefolded


def test_parse_voice_overrides_bare_last_wins_and_empty_default():
    assert dialogue.parse_voice_overrides([]) == (None, {})
    assert dialogue.parse_voice_overrides(["jane", "mary"]) == ("mary", {})


@pytest.mark.parametrize("bad", ["=vera", "A=", "  =  "])
def test_parse_voice_overrides_rejects_malformed_pair(bad: str) -> None:
    with pytest.raises(UsageError):
        dialogue.parse_voice_overrides([bad])


def test_assign_voices_rotates_in_first_appearance_order():
    segs = [dialogue.Segment(s, "x") for s in ("A", "B", "A", "C")]
    resolved, mapping = dialogue.assign_voices(segs, ["jane", "michael", "mary"], {})
    assert [v for v, _ in resolved] == ["jane", "michael", "jane", "mary"]
    assert mapping == {"A": "jane", "B": "michael", "C": "mary"}


def test_assign_voices_wraps_past_rotation_length():
    # 3-voice rotation, 4 speakers: the 4th wraps to the 1st voice. This only holds
    # when the rotation index advances correctly, so it pins the wrap arithmetic.
    segs = [dialogue.Segment(s, "x") for s in ("A", "B", "C", "D")]
    resolved, _ = dialogue.assign_voices(segs, ["jane", "michael", "mary"], {})
    assert [v for v, _ in resolved] == ["jane", "michael", "mary", "jane"]


def test_assign_voices_override_beats_rotation_without_consuming_a_slot():
    segs = [dialogue.Segment(s, "x") for s in ("A", "B")]
    # A is overridden, so B still gets the FIRST rotation voice, not the second.
    resolved, mapping = dialogue.assign_voices(segs, ["jane", "michael"], {"a": "vera"})
    assert [v for v, _ in resolved] == ["vera", "jane"]
    assert mapping == {"A": "vera", "B": "jane"}


def test_default_rotation_is_the_confirmed_working_voices():
    assert dialogue.DEFAULT_VOICE_ROTATION == ("jane", "michael", "mary", "paul", "eve", "george")
