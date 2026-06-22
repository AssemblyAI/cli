# Live voice agent: speak a filler while a tool runs

Date: 2026-06-22
Branch: `live-tool-call-ux`
Status: design (awaiting review)

## Problem

`assembly live` (the `agent_cascade`) answers spoken turns with a deepagents
graph that can pause mid-turn to call a tool (`get_weather`, Firecrawl web
search, or an MCP tool). While that tool runs, the cascade emits a **visual**
affordance only — `renderer.tool_call("Searching the web")` mounts a dim note in
the TUI — but **says nothing audibly**. On a hands-free voice session the user
hears dead air for the whole tool round-trip and assumes the agent broke or
didn't hear them.

This is the single highest-impact responsiveness fix borrowed from OpenClaw,
whose realtime voice agent speaks a brief "let me check" before delegating work
(`buildRealtimeVoiceAgentConsultWorkingResponse`).

## Goal

When the agent starts its first tool call of a turn, speak a short, spoken-style
filler ("Let me check the weather", "Let me look that up") through the existing
TTS leg, so the silent gap is filled and the user knows work is happening.

Non-goals:

- No change to the LLM prompt or to how tools are selected/called.
- No new CLI flag or config field (v1 ships always-on).
- No streaming-TTS-only restriction beyond what already gates the cascade.

## Design decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Filler source | **Canned per-tool phrases**, keyed off the existing `_TOOL_LABELS` map. Deterministic, zero extra LLM latency, fully testable, and it says *why* the agent paused. |
| When to speak | **First tool call of a turn only.** Chained tool calls stay silent so a multi-tool turn doesn't get chatty. |
| Variety | **Rotate 2–3 phrases per tool deterministically** by a per-session counter (not RNG — so it's testable and survives the mutation gate). |
| Configurable? | **No.** Always-on in v1; a toggle can be added later if anyone wants silence. |

## Where it lives

The seam already exists. In `aai_cli/agent_cascade/engine.py`, `_consume`
already receives a `brain.ToolNotice` for each started tool call:

```python
if isinstance(item, brain.ToolNotice):
    self.renderer.tool_call(item.label)
    buffer = ""  # drop any unspoken preamble — the answer comes after the tool
    continue
```

The filler hooks in right here: after showing the visual affordance, synthesize
a spoken filler through the same path a normal clause uses (`_speak` →
`synthesize` → `_feed` → `player`), so barge-in (`_stop`) and the draining-tail
logic already cover it for free.

The phrase table lives in `aai_cli/agent_cascade/brain.py` next to
`_TOOL_LABELS`, because the filler is a property of the tool (same place we
already keep the human-readable label) and `ToolNotice` is the natural carrier.

### Components

1. **`brain.py` — filler phrases + carrier.**
   - Add a `_TOOL_FILLERS: dict[str, tuple[str, ...]]` mapping each known tool
     name to a small tuple of spoken variants, plus a generic fallback tuple
     (e.g. `("One sec.", "Let me check.")`) for unknown/MCP tools.
     - `WEB_SEARCH_TOOL_NAME` → e.g. `("Let me look that up.", "Searching now.", "One moment, checking the web.")`
     - `weather_tool.WEATHER_TOOL_NAME` → e.g. `("Let me check the weather.", "Checking the forecast now.")`
   - Carry the chosen filler on `ToolNotice`. Extend the dataclass with a
     `fillers: tuple[str, ...]` field (the variants for that tool), set when the
     notice is built in `_events_from_chunk` / `_surface_event` via a new
     `_tool_fillers(name)` helper that mirrors `_tool_label(name)`.
   - Keeping the *tuple* on the notice (not a pre-chosen single string) lets the
     engine own rotation state, so two notices for the same tool in one session
     rotate rather than repeat. The notice stays a pure value object.

2. **`engine.py` — speak it once per turn, rotate across turns.**
   - Add a per-session rotation counter to `CascadeSession`
     (`_filler_index: int`, init `0`, `# pragma: no mutate` on the field if a
     ±-equivalent default trips the gate).
   - Add a per-turn `spoke_filler: bool` guard local to `_consume` so only the
     **first** `ToolNotice` of a turn speaks. (Track it as a local, reset each
     `_consume` call.)
   - On the first `ToolNotice`: pick `fillers[self._filler_index % len(fillers)]`,
     increment `_filler_index`, and synthesize it via the existing `_speak`
     machinery so it respects `_stop` and feeds the player. The filler text is
     **not** appended to `spoken`/history — it is conversational glue, not part
     of the answer (history must stay a clean alternating record of the real
     reply). This means routing the filler through `synthesize`/`_feed`
     directly, or a thin `_speak_filler(text)` that mirrors `_speak` but skips
     the `spoken.append`.
   - `started`/`reply_started` handling: the filler counts as the start of
     audible output, so set `_speaking`/call `reply_started()` before
     synthesizing the filler if not already started (same as a normal clause),
     so the voice bar shows "speaking" and a barge-in during the filler is
     detected.

### Data flow

```
graph stream → ToolNotice(label, fillers)        (brain.py)
  → engine._consume sees first ToolNotice of turn
      → renderer.tool_call(label)         # existing visual affordance
      → _speak_filler(pick(fillers))      # NEW: spoken filler, not recorded
          → synthesize(text, _feed) → player.enqueue   # respects _stop
      → buffer = ""                       # existing: drop preamble
  → subsequent ToolNotices in same turn: visual only (no filler)
  → real answer clauses stream in and are spoken + recorded as today
```

## Interruption / barge-in

The filler rides the same `_stop` / `player.flush()` path as any clause:

- A spoken barge-in (`on_turn` → `_barge_in`) sets `_stop` and flushes queued
  audio, so a filler mid-playback is cut just like a reply clause.
- A UI interrupt (`interrupt_reply`) flushes the player; since the filler will
  have set `_speaking`, the interrupt is detected (not swallowed as a no-op).
- `_feed` already drops frames once `_stop` is set, so a filler can't keep
  playing after the user barges in.

No new interruption logic is needed. (Echo-induced *false* barge-in — the mic
hearing the filler/agent audio — is a **separate** problem tracked in the echo
guard spec; this spec does not address it.)

## Error handling

- If `synthesize` raises `CLIError` on the filler, reuse `_speak`'s existing
  contract: record the error and stop the turn (return as a cut). A filler that
  can't be synthesized is the same failure mode as a clause that can't —
  surfaced once, turn ends cleanly. The real-answer path is unaffected.
- An unknown tool name (no entry in `_TOOL_FILLERS`) falls back to the generic
  filler tuple, exactly as `_tool_label` falls back to `"Using {name}"`.

## Testing

The cascade is unit-tested against fakes through `CascadeDeps` (no
sockets/mic/speaker). New coverage, all driving the fake `stream_reply`/
`synthesize`:

1. **Filler is spoken on first tool call.** Script a `stream_reply` that yields
   `ToolNotice` then `SpeechDelta`s; assert the fake `synthesize` received a
   filler string from the tool's tuple *before* the answer clauses. (Kills a
   mutant that drops the filler call.)
2. **Only the first tool call speaks.** Yield two `ToolNotice`s in one turn;
   assert exactly one filler was synthesized. (Kills a mutant that removes the
   `spoke_filler` guard.)
3. **Rotation across turns.** Run two turns that each trigger the same tool;
   assert the two fillers differ (index advanced). (Kills a mutant that pins the
   index to 0.)
4. **Filler is not in history.** After a tool turn, assert `session.history`'s
   assistant message is the real answer only — no filler text. (Kills a mutant
   that appends the filler to `spoken`.)
5. **Barge-in cuts the filler.** Set `_stop` (or drive an interim turn) during
   filler synthesis; assert no further frames are fed. (Reuses the existing
   barge-in test harness.)
6. **Unknown/MCP tool uses the generic fallback.** A `ToolNotice` for a tool not
   in `_TOOL_FILLERS` still speaks a generic filler.

Per the repo gate, every new line needs an assertion that *fails* if the line
breaks (mutation gate), and the diff needs 100% patch coverage.

## Risks / open questions

- **Phrase wording** is a copy decision; the tuples above are placeholders to
  refine. They must obey the spoken-style rule (short, no markdown) and read
  naturally before the real answer.
- **Latency interaction:** the filler adds one extra TTS round-trip before the
  answer. Because synthesis streams (playback starts on the first frame) and the
  tool call is already the slow leg, the filler should overlap the tool
  round-trip rather than serialize behind it — but verify the filler doesn't
  noticeably delay the first answer clause in a real sandbox run.
- **MCP tools** get a generic filler; once MCP tools are common we may want
  per-tool fillers derived from the tool description, but that is out of scope.

## Out of scope

- Echo / false-barge-in suppression (separate spec).
- Model-emitted acknowledgements via prompt.
- A config flag / `--no-tool-filler` toggle.
