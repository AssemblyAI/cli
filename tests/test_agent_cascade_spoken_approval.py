"""Tests for the spoken-approval grammar (`assembly live --files` hands-free gate)."""

from __future__ import annotations

import pytest

from aai_cli.agent_cascade.spoken_approval import interpret_spoken_approval, resolve_approval


def _resolve(name, args, *, outcome, keyboard):
    return resolve_approval(name, args, keyboard=keyboard, await_outcome=lambda: outcome)


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


def test_resolve_benign_voice_affirmative_approves():
    assert (
        _resolve(
            "write_file", {"file_path": "n.txt"}, outcome=("voice", "yes, run it"), keyboard=_unused
        )
        is True
    )


def test_resolve_benign_voice_bare_yes_rejects():
    # A bare "yes" must not approve even on the voice channel (fail-safe).
    assert (
        _resolve("write_file", {"file_path": "n.txt"}, outcome=("voice", "yes"), keyboard=_unused)
        is False
    )


def test_resolve_benign_voice_negative_rejects():
    assert (
        _resolve("write_file", {"file_path": "n.txt"}, outcome=("voice", "no"), keyboard=_unused)
        is False
    )


def test_resolve_benign_keypress_is_taken_verbatim():
    assert (
        _resolve("write_file", {"file_path": "n.txt"}, outcome=("key", True), keyboard=_unused)
        is True
    )
    assert (
        _resolve("write_file", {"file_path": "n.txt"}, outcome=("key", False), keyboard=_unused)
        is False
    )


def test_resolve_benign_timeout_rejects():
    assert (
        _resolve("write_file", {"file_path": "n.txt"}, outcome=("timeout", None), keyboard=_unused)
        is False
    )


def test_resolve_destructive_ignores_voice_and_requires_keyboard():
    # A destructive command (risk.risk_warning fires) must IGNORE a spoken affirmative and resolve
    # via the keyboard only — an STT mishearing can never green-light it.
    calls: list[tuple[str, dict]] = []

    def keyboard(name, args):
        calls.append((name, args))
        return False  # the human declines at the keyboard

    voiced_approve = ("voice", "approve")  # would approve if voice were honored
    decided = resolve_approval(
        "execute",
        {"command": "rm -rf build"},
        keyboard=keyboard,
        await_outcome=lambda: voiced_approve,
    )
    assert decided is False  # keyboard's decision, not the spoken "approve"
    assert calls == [("execute", {"command": "rm -rf build"})]  # keyboard was consulted


def _unused(name, args):  # the keyboard must not be consulted on the benign (voice/key) paths
    raise AssertionError("keyboard should not be called on the non-destructive path")
