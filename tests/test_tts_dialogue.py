from __future__ import annotations

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
