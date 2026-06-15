"""Unit tests for the streaming turn-detection presets (aai_cli.streaming.turn_presets).

The presets mirror the documented Aggressive/Balanced/Conservative quick-start
configurations (streaming/universal-streaming/turn-detection). `resolve` merges a
preset with explicitly-passed raw flags, where an explicit value always wins.
"""

from __future__ import annotations

import pytest

from aai_cli.streaming import turn_presets
from aai_cli.streaming.turn_presets import TurnDetectionPreset


def test_no_preset_passes_raw_values_through_unchanged():
    assert turn_presets.resolve(None, None, None, None) == (None, None, None)
    assert turn_presets.resolve(None, 0.5, 300, 900) == (0.5, 300, 900)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (TurnDetectionPreset.aggressive, (0.4, 160, 400)),
        (TurnDetectionPreset.balanced, (0.4, 400, 1280)),
        (TurnDetectionPreset.conservative, (0.7, 800, 3600)),
    ],
)
def test_preset_supplies_documented_values(preset, expected):
    assert turn_presets.resolve(preset, None, None, None) == expected


def test_explicit_min_turn_silence_overrides_only_its_slot():
    # balanced is (0.4, 400, 1280); overriding min_turn_silence keeps the other two.
    assert turn_presets.resolve(TurnDetectionPreset.balanced, None, 500, None) == (0.4, 500, 1280)


def test_explicit_confidence_overrides_preset_confidence():
    # conservative is (0.7, 800, 3600); an explicit eot threshold wins.
    assert turn_presets.resolve(TurnDetectionPreset.conservative, 0.9, None, None) == (
        0.9,
        800,
        3600,
    )


def test_all_explicit_flags_override_every_preset_slot():
    assert turn_presets.resolve(TurnDetectionPreset.aggressive, 0.1, 50, 100) == (0.1, 50, 100)
