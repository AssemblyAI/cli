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

# One resolution of the race the engine runs during an approval pause: the keypress decision
# (``"key"``, a bool), the next spoken transcript (``"voice"``, the text), or nothing in the
# window (``"timeout"``). The engine supplies the racing implementation; tests inject outcomes.
Outcome = tuple[str, object]
AwaitOutcome = Callable[[], Outcome]
Keyboard = Callable[[str, dict[str, object]], bool]

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


def resolve_approval(
    name: str,
    args: Mapping[str, object],
    *,
    keyboard: Keyboard,
    await_outcome: AwaitOutcome,
    warn: Callable[[str, Mapping[str, object]], str | None] = risk.risk_warning,
) -> bool:
    """Resolve one ``--files`` approval, voice-or-keyboard, fail-safe to reject.

    Destructive tier (``risk.risk_warning`` fires) → the spoken channel is ignored and the
    keyboard is required, so an STT mishearing can never green-light an ``rm -rf``/``sudo``.
    Otherwise the engine's race (``await_outcome``) resolves it: a keypress is taken verbatim, a
    spoken transcript is run through :func:`interpret_spoken_approval`, and a timeout — like any
    ambiguous or negative answer — rejects.
    """
    if warn(name, args) is not None:
        return keyboard(name, dict(args))
    kind, value = await_outcome()
    if kind == "key":
        return bool(value)
    if kind == "voice":
        return interpret_spoken_approval(str(value))
    return False
