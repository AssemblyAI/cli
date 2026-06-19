"""Unit tests for the shared media-file helpers in `aai_cli.app.mediafile`.

The source-resolution, ffmpeg, and transcript-fetch helpers are exercised through
the clip/dub/caption command tests; here we pin the small standalone helpers that
several commands route through, so a refactor can't silently change their contract.
"""

from __future__ import annotations

from aai_cli.app import mediafile


class _Transcript:
    def __init__(self, transcript_id: object) -> None:
        self.id = transcript_id


def test_transcript_id_returns_the_id_as_a_string():
    assert mediafile.transcript_id(_Transcript("tr_123")) == "tr_123"


def test_transcript_id_is_empty_when_the_object_carries_no_id():
    assert mediafile.transcript_id(object()) == ""
