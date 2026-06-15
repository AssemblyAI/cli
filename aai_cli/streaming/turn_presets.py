"""Documented turn-detection quick-start presets for `assembly stream`.

The Aggressive/Balanced/Conservative configurations mirror the streaming
turn-detection docs (streaming/universal-streaming/turn-detection). A preset
sets the three end-of-turn knobs together; `resolve` lets any explicitly-passed
raw flag override its slot so users can start from a preset and tweak one value.
"""

from __future__ import annotations

import enum


class TurnDetectionPreset(enum.StrEnum):
    """Named end-of-turn sensitivity presets from the streaming turn-detection docs."""

    aggressive = "aggressive"
    balanced = "balanced"
    conservative = "conservative"


# (end_of_turn_confidence_threshold, min_turn_silence, max_turn_silence) per the docs'
# quick-start configurations. Keep these verbatim — they're the published recommendations.
_PRESETS: dict[TurnDetectionPreset, tuple[float, int, int]] = {
    TurnDetectionPreset.aggressive: (0.4, 160, 400),
    TurnDetectionPreset.balanced: (0.4, 400, 1280),
    TurnDetectionPreset.conservative: (0.7, 800, 3600),
}


def resolve(
    preset: TurnDetectionPreset | None,
    end_of_turn_confidence_threshold: float | None,
    min_turn_silence: int | None,
    max_turn_silence: int | None,
) -> tuple[float | None, int | None, int | None]:
    """Merge a preset with raw flags, where an explicitly-passed value wins its slot.

    With no preset the three values pass through unchanged (server defaults apply).
    """
    if preset is None:
        return end_of_turn_confidence_threshold, min_turn_silence, max_turn_silence
    preset_eot, preset_min, preset_max = _PRESETS[preset]
    return (
        end_of_turn_confidence_threshold
        if end_of_turn_confidence_threshold is not None
        else preset_eot,
        min_turn_silence if min_turn_silence is not None else preset_min,
        max_turn_silence if max_turn_silence is not None else preset_max,
    )
