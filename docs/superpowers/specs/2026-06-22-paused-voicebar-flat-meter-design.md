# Paused voice-bar: flat (non-animating) meter

**Date:** 2026-06-22
**Status:** Approved (design)
**Scope:** The shared voice-bar helper used by the `assembly live` and `assembly code` TUIs.
**Depends on:** the in-flight push-to-talk work (the `"paused"` phase), which is uncommitted in the main checkout — not on `origin/main`, so this is **not** part of PR #258.

## Problem

The push-to-talk work added a `"paused"` voice phase (`tui_status.py:30`) for a muted mic. But the bar's meter keeps **animating** while paused: `_render_voicebar` (`agent_cascade/tui.py:271`) always passes `next(self._voice_frames)`, so the 3-cell block pulse (`▁▃▅`→`▃▅▇`→…) cycles every 0.3s even though nothing is being heard. A paused session should read as at-rest, not active.

## Design

Flatten the meter inside the shared pure helper `tui_status.voicebar_markup`, so the `"paused"` phase renders a static at-rest meter regardless of the frame it is handed:

```python
# at-rest meter for the paused phase (same width/alphabet as VOICE_FRAMES)
VOICE_FLAT = "▁▁▁"

def voicebar_markup(phase: str, frame: str, *, hint: str = "") -> str:
    label, color = _VOICE_PHASES[phase]
    if phase == "paused":
        frame = VOICE_FLAT  # a muted mic shows a flat meter, not the animated pulse
    return f"[{color}]{frame}[/] {escape(label)}{hint}"
```

**Why in the helper (not the caller):**
- `voicebar_markup` is a pure function (no Textual), so the behavior unit-tests directly with no app/timer.
- The helper is shared by both the `live` and `code` TUIs, so both surfaces get the flat paused meter from one change.
- `_render_voicebar` keeps calling `next(self._voice_frames)` unchanged — the animation cycle advances invisibly while paused, so there is no timer or iterator state to manage. The displayed meter is simply static.

Rejected alternatives: a conditional in each TUI's `_render_voicebar` (caller-side, duplicated across both TUIs); stopping/restarting the 0.3s animation timer on pause (manages timer lifecycle state for no visible benefit over a static frame).

## Components touched

- `aai_cli/code_agent/tui_status.py` — add `VOICE_FLAT`; add the paused-frame override in `voicebar_markup`.

No change to `agent_cascade/tui.py`, `code_agent/tui.py`, `_VOICE_PHASES`, or `VOICE_FRAMES`.

## Testing

Update the existing `tests/test_code_tui_status.py::test_voicebar_markup_per_phase_carries_label_meter_accent_and_hint`:
- Assert the paused render contains the **literal** `"▁▁▁"` (not `tui_status.VOICE_FLAT`, which would mutate in lockstep and survive — per the file's existing comment at lines 45–46) and does **not** contain the animated frame it was passed (`▁▃▅`) — this kills the mutant on the new `if phase == "paused"` branch and on the `VOICE_FLAT` literal.
- The existing non-paused assertions already prove the frame passes through verbatim for `listening`/`thinking`/`speaking` (e.g. `"▁▃▅" in listening`), guarding against a mutant that flattens every phase.

No visual snapshot exists for the paused state, so there is nothing to regenerate.

## Landing

Implement in the main checkout, alongside the in-flight push-to-talk work (where the `"paused"` phase lives). Run only the targeted `test_code_tui_status.py` tests — the full gate is not run here because the main checkout's working tree carries unrelated, half-finished in-flight work. Do **not** commit or modify the rest of the in-flight changes; this edit joins that work for the user to gate and PR as a unit.

## Out of scope

- Stopping the animation timer while paused (the meter only needs to *look* static).
- Any change to the `assembly code` voice chrome beyond what the shared helper provides for free.
