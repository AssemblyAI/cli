# Spoken approval for `assembly live --files` (M3) Implementation Plan

> **For agentic workers:** superpowers:subagent-driven-development / :executing-plans.

**Goal:** Let the `--files` approval gate be answered by an unambiguous **spoken** yes/no (not only a keypress), so the safety gate doesn't contradict the hands-free premise — with a keyboard fallback for the highest-risk (destructive) commands.

**Spec:** `docs/superpowers/specs/2026-06-22-live-sandboxed-execute-design.md` (Milestone **M3** — "the largest lift, touching engine/modals"). Builds on M1 (gated execute) + M2 (subagents), committed.

## Status
- **DONE — token grammar (the safety core):** `aai_cli/agent_cascade/spoken_approval.py` + tests (commit `c68703c`). `interpret_spoken_approval(transcript) -> bool` is **fail-safe to reject**: only an unambiguous action-bearing affirmative ("approve" / "yes, run it" / "go ahead and run it") returns True; a bare "yes", any negation, unrelated/empty speech all return False.
- **REMAINING — the engine STT-vs-keypress race + destructive-tier keyboard fallback** (below). Touches `engine.py` / `tui.py` / `_exec.py`, which a concurrent session is actively rewriting — land once that settles to avoid building on a moving base.

## Architecture of the remaining work

Today the `Approver` (`brain.Approver = Callable[[str, dict], bool]`) is invoked **synchronously on the cascade worker thread** inside `brain._stream_gated`, bracketed by `ApprovalPause(active=True/False)` (so `engine._consume` suspends the reply deadline). The TUI supplies it via `app.approve_write(name, args)` (blocks on `modals.ApprovalScreen`'s keypress); headless uses `_exec._deny_writes`. Spoken approval makes the *answer source* multimodal without changing the gate's shape.

### Task A — a voice-aware approver the engine supplies (`engine.py` + an injected token source)
- The engine owns the STT leg (`run_stt(on_turn)`); during an `ApprovalPause(active=True)` it must capture the **next final transcript** and offer it to the approval decision, racing a keypress.
- Add an injectable **spoken-token source** seam: `Callable[[float], str | None]` — "wait up to `timeout` s for the next final transcript, or None". The production impl reads from the live STT leg (a queue the `on_turn` final-transcript path feeds during a pause); tests inject a fake that returns a scripted phrase or None — **no mic, no sockets** (mirrors the existing `CascadeDeps` fakes).
- The voice-aware approver: when invoked, it (1) consults `risk.risk_warning(name, args)` — if it fires (destructive tier), **ignore voice, require the keyboard** (delegate to the existing keypress approver); else (2) races the spoken-token source against the keypress, resolving with whichever lands first: a spoken token → `interpret_spoken_approval(token)`; a keypress → its decision; timeout/None/ambiguous → reject (the existing `_DECLINED` path).
- Keep `modals.ApprovalScreen` (keypress) as the fallback and the floor for the destructive tier; `_deny_writes` (headless) unchanged.

### Task B — wire it through `_exec.py` / `tui.py`
- The TUI run currently passes `approver=approve_write` (keypress). Wrap it in the voice-aware approver, handing it the spoken-token source (from the engine's STT leg) and the keypress approver as the fallback. The destructive-tier branch routes to `approve_write` (keyboard) directly.
- Surface the spoken-vs-keyboard affordance on `ApprovalScreen` copy ("say 'approve' or press y") — regenerate the TUI snapshot if the modal chrome changes.

### Task C — tests (hermetic, via the injected seams)
- An explicit affirmative phrase from the fake token source approves; a bare "yes", a negative, an unrecognized utterance, and a timeout each reject; a keypress still approves; and a `risk.py`-flagged destructive command **ignores** the spoken affirmative and requires the keypress. Assert the *resolved decision*, not mere execution (kills the mutation-gate mutants on the race/risk branches).
- `risk.py`'s destructive branch is exercised by the destructive-tier test (it's already covered by `tests/test_live_risk.py`).

## Constraints (carry from M1/M2)
- Hermetic: inject the spoken-token source + keypress; no mic/sockets. `from __future__ import annotations`; no new dependency.
- Fail-safe to reject is the invariant: every non-clear-affirmative path → reject.
- 100% patch coverage + mutation gate; no new escape hatches; tests-pyright via `-p pyrightconfig.tests.json` (add the file to its ignore list if it builds real deepagents graphs).
- Concurrent session churns `engine.py`/`tui.py`: commit only M3 files; `AAI_ALLOW_COMMIT=1` per task; final `./scripts/check.sh` (sandbox-disabled).
