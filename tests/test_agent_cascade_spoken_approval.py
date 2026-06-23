"""Tests for the spoken-approval grammar (`assembly live --files` hands-free gate)."""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade.spoken_approval import interpret_spoken_approval


@pytest.mark.parametrize(
    "transcript",
    [
        "approve",
        "Approve.",
        "yes, run it",
        "run it",
        "go ahead and run it",
        "go ahead",
        "do it",
        "yeah, go for it",
    ],
)
def test_explicit_affirmatives_approve(transcript: str) -> None:
    assert interpret_spoken_approval(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "yes",  # bare yes never approves (STT mishears it)
        "yeah",
        "sure",
        "okay",
        "no",
        "no, don't run it",  # negation wins even though it contains "run it"
        "stop",
        "cancel that",
        "do not run it",
        "what's the weather",  # unrelated utterance
        "",  # silence / empty final transcript
    ],
)
def test_non_affirmatives_reject(transcript: str) -> None:
    assert interpret_spoken_approval(transcript) is False
