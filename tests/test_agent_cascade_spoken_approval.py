"""Tests for the spoken-approval grammar (`assembly live --files` hands-free gate)."""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade.spoken_approval import interpret_spoken_approval, spoken_decision


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


def test_spoken_decision_benign_affirmative_approves():
    assert spoken_decision("write_file", {"file_path": "n.txt"}, "yes, run it") is True


def test_spoken_decision_benign_bare_yes_rejects():
    # A bare "yes" must not approve even on the voice channel (fail-safe).
    assert spoken_decision("write_file", {"file_path": "n.txt"}, "yes") is False


def test_spoken_decision_benign_negative_rejects():
    assert spoken_decision("write_file", {"file_path": "n.txt"}, "no") is False


def test_spoken_decision_destructive_ignores_voice():
    # A destructive command (risk.risk_warning fires) returns None — the spoken channel is ignored
    # even for an explicit "approve", so only the keyboard can green-light it.
    assert spoken_decision("execute", {"command": "rm -rf build"}, "approve") is None
    assert spoken_decision("execute", {"command": "sudo make install"}, "yes, run it") is None


def test_spoken_decision_execute_is_always_keypress_only():
    # Running code is never voice-approvable, even a benign command with an explicit affirmative:
    # a misheard "go ahead" must not run arbitrary code, so execute always returns None.
    assert spoken_decision("execute", {"command": "pytest -q"}, "go ahead") is None
    assert spoken_decision("execute", {"command": "ls -la"}, "approve") is None


def test_spoken_decision_benign_file_write_honors_voice():
    # A non-execute write (sandbox-contained, git-recoverable) still takes the spoken decision.
    assert spoken_decision("write_file", {"file_path": "n.txt"}, "go ahead") is True
    assert spoken_decision("edit_file", {"file_path": "n.txt"}, "no") is False
