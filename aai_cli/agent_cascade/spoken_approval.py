"""Spoken-approval grammar for ``assembly live --files`` (the hands-free approval gate).

During a write/run approval pause the engine may answer the gate with the user's *next spoken
transcript* instead of a keypress. STT is noisy and a mis-heard "yes" must never green-light a
mutation, so this grammar is **fail-safe to reject**: only an unambiguous, action-bearing
affirmative ("approve", "yes, run it", "go ahead and run it") counts as approval. A bare "yes",
any negation, an unrelated utterance, or empty text all read as reject. Pure functions so they
unit-test cleanly; the risk-tier keyboard fallback (destructive commands need a keypress) lives
in the engine, which consults ``risk.py`` before trusting a spoken yes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from aai_cli.agent_cascade import risk

# A negation anywhere flips the whole utterance to reject — so "no, don't run it" can't approve
# just because it contains "run it". Checked first, before the affirmative patterns.
_NEGATION = re.compile(r"\b(no|nope|don'?t|do not|stop|cancel|never|reject|deny|wait)\b", re.I)

# Unambiguous, action-bearing affirmatives. Deliberately excludes bare "yes"/"yeah"/"sure"/"ok",
# which STT confuses with "no"/"go" — approval must carry an explicit action or the word "approve".
_AFFIRMATIVE = re.compile(r"\b(approve|approved|run it|do it|go ahead|go for it)\b", re.I)


def interpret_spoken_approval(transcript: str) -> bool:
    """True only for an unambiguous spoken approval; everything else is False (fail-safe reject).

    Rejects on any negation, on a bare "yes" (no action word), on unrelated/empty speech — so a
    mis-heard token can never approve a mutation. A genuine affirmative ("approve", "yes, run it",
    "go ahead and run it") with no negation returns True.
    """
    text = transcript or ""
    if _NEGATION.search(text):
        return False
    return bool(_AFFIRMATIVE.search(text))


def spoken_decision(
    name: str,
    args: Mapping[str, object],
    transcript: str,
    *,
    warn: Callable[[str, Mapping[str, object]], str | None] = risk.risk_warning,
) -> bool | None:
    """How a spoken transcript should resolve an open approval: True approve, False reject, or
    None *ignore the voice* (the destructive tier — require the keyboard).

    Destructive tier (``risk.risk_warning`` fires, e.g. ``rm -rf``/``sudo``) → None, so an STT
    mishearing can never green-light it; the keypress is the only channel. Otherwise the grammar
    decides: an unambiguous affirmative approves, everything else rejects (fail-safe).
    """
    if warn(name, args) is not None:
        return None
    return interpret_spoken_approval(transcript)
