# Live voice agent: echo guard against the agent's own TTS (half-duplex)

Date: 2026-06-22
Branch: `live-tool-call-ux`
Status: design (awaiting review)

## Problem

`assembly live` (the `agent_cascade`) runs mic + speaker through a single
full-duplex PortAudio stream (`DuplexAudio`, `aai_cli/agent/audio.py`). The mic
stays open while the agent speaks, and `CascadeSession.on_turn` barges in on
**any** non-empty interim transcript. On laptop speakers the mic hears the
agent's own TTS, STT transcribes it, and the agent **interrupts itself** — it
cuts its reply off mid-sentence as if the user spoke.

The team already knows this: `_exec._open_audio` prints

> "Use headphones — the mic stays open while the agent speaks, so speakers would
> let it hear itself."

That headphones warning is the *current* mitigation. This spec replaces it with
a real guard so speakers work.

We confirmed OpenClaw does **not** implement acoustic echo cancellation (AEC):
its browser path delegates `echoCancellation` to the OS via `getUserMedia`
constraints, and its native realtime path leans on server VAD plus a
`minBargeInAudioEndMs` debounce. Our raw `sd.RawStream` gets neither for free,
so AEC is not prior art to copy — a half-duplex gate is the robust option.

## Goal

While the agent is producing audio, feed **silence** to STT so it never
transcribes the agent's own voice and can't trigger a self-barge-in. Re-open the
mic the moment the agent's audio finishes draining.

Non-goals / accepted tradeoff:

- **Voice "talk-over" barge-in is disabled while the agent speaks.** The user
  cannot interrupt by voice mid-reply; they interrupt with the existing UI
  control (Escape / Ctrl-C → `interrupt_reply`), which flushes playback, drains
  the buffer, and re-opens the mic so voice resumes immediately. This is the
  explicit, chosen consequence of half-duplex.
- No AEC, no DSP, no platform-specific audio APIs.
- No CLI flag / config field — on by default (v1).

## Design decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Strategy | **Half-duplex mic gate during playback.** Mute capture (feed silence to STT) for the whole speaking phase and until the playback buffer drains. |
| Configurable? | **No.** Always-on in v1. A `--full-duplex` toggle for headphone users who want talk-over barge-in can come later. |
| Clock / timing | **None.** Gate on engine speaking-phase state + buffer-non-empty, so it's deterministic and clock-free (no flaky tail timer). |

## Why half-duplex, and why scoped to the cascade

`DuplexAudio` is shared with `assembly agent` (the Voice Agent endpoint path,
`agent/session.py`). The guard must not change that path. It won't, because the
guard is **driven by the consumer**: only `engine.run_cascade` calls the new
"output active" hooks. `assembly agent` runs `run_session`, never touches them,
so its behavior is unchanged with no opt-in flag needed on `DuplexAudio`.

## Components

### 1. `aai_cli/agent/audio.py` — a second, independent mic gate

`DuplexAudio` already has one mic gate: `_listening` (the user's Space-to-mute).
Add a **second, independent** gate for the echo guard so the two never clobber
each other:

- New `self._output_active: threading.Event` (clear by default), with
  `set_output_active(on: bool)`.
- `capture_frames` feeds silence to STT when **any** of these hold (compose with
  the existing `_listening` check):

  ```text
  mute if   (not self._listening.is_set())          # user muted (existing)
         or  self._output_active.is_set()            # engine: speaking phase
         or  len(self._out) > 0                       # NEW: playback buffer draining
  ```

  - `_output_active` covers the **whole** speaking phase, including the silent
    inter-clause gaps while the next clause is still synthesizing (so the mic
    doesn't flicker open between sentences).
  - `len(self._out) > 0` (the same quantity `pending()` exposes) covers the
    **drain tail** after the engine clears `_output_active`: the last clause's
    audio is still in the buffer, so the mic stays muted until it empties — no
    clock needed.
  - Existing `_listening` keeps the user's manual mute authoritative and
    composes (mic is live only when listening **and** no output activity).

  The silence is produced exactly as the existing muted path does
  (`chunk = bytes(len(chunk))` before resample), so the STT socket stays alive
  and reconnect-free — a proven path (it's how Space-to-mute already works).

`set_output_active` flips an `Event`, so it's safe to call from the engine /
reply-worker threads while `capture_frames` reads it on the capture thread (same
pattern as `_listening`).

### 2. `engine.py` Player protocol — `begin_output()` / `end_output()`

The engine's only handle to the duplex device is the `Player` protocol (it
receives `duplex.player`; the mic is hidden inside `deps.run_stt`'s audio
iterable). Extend the protocol:

- `begin_output()` — called when audible output starts.
- `end_output()` — called when the engine has finished enqueuing the last frame
  of the speaking phase.

Implementations:

- `_DuplexPlayer` (audio.py) delegates to its `DuplexAudio.set_output_active`
  (`begin_output` → `set_output_active(on=True)`, `end_output` →
  `set_output_active(on=False)`).
- `NullPlayer` (file-driven / headless) — no-ops (no live mic; `pending()` is
  always 0, so the drain term is moot there too).
- The cascade test fake player gains the two no-op/recording methods.

### 3. `engine.py` `CascadeSession` — drive the hooks

Mute around every audible phase, leaving the drain term to cover the tail:

- **Greeting** (`greet`): `player.begin_output()` before `synthesize(greeting,
  …)`, `player.end_output()` after it returns. The greeting audio keeps the mic
  muted while it drains via the buffer term.
- **Reply** (`_consume` / `_generate_reply`): call `begin_output()` at the point
  `started` flips True (the first audible clause — same place `reply_started()`
  fires), and `end_output()` in the reply's teardown alongside
  `_speaking.clear()` / `reply_done(...)`, so it runs on every exit path (clean
  finish, barge-in, TTS/leg failure, timeout).
- **Thinking / tool calls produce no audio**, so the mic stays open during them
  — the user can still speak while the agent thinks or runs a tool (no echo to
  guard against there). Muting is strictly tied to *audible output*.

`on_turn` is **unchanged**: with the mic fed silence during playback, STT simply
emits no interim/final turns then, so the existing barge-in code never fires on
echo. The fix is localized to the audio layer plus the two engine hooks.

### 4. `_exec.py` — relax the headphones warning

`_open_audio`'s notice (lines 168-171) becomes accurate for the new behavior,
e.g.:

> "Speakers are fine — the mic mutes while the agent speaks. To interrupt it,
> press Esc (talking over it won't cut in)."

Update any test asserting the old copy. The voice-only TUI's listen indicator
(driven by the user's `toggle_listening`) is unaffected — the echo-guard gate is
a separate `Event`, invisible to the manual mute state.

## Data flow

```
greet():            begin_output() → synthesize(greeting) → end_output()
                       └─ mic muted (output_active) … then muted while _out drains … reopens

reply turn:
  thinking / tool:  mic OPEN (no audio; user may speak)
  first clause:     begin_output() + reply_started()
  clauses stream:   synthesize → player.enqueue;  mic muted (output_active) across inter-clause gaps
  last clause done: end_output();  mic stays muted while _out drains, then reopens
  barge-in (Esc):   interrupt_reply → flush() (_out cleared) ; worker teardown → end_output()
                       └─ _out empty + output_active cleared → mic reopens at once → voice resumes
```

## Interruption semantics (the tradeoff, stated plainly)

- **During playback:** voice over-talk does nothing (STT hears silence). Esc /
  Ctrl-C (`interrupt_reply`) cuts the agent off; it flushes the buffer and clears
  output-active, so the mic reopens immediately and the user can speak.
- **After playback drains:** normal turn handling resumes; the next spoken turn
  is detected and answered as today.
- A spoken barge-in that arrives in the gap *after* the agent finishes (mic
  already reopened) works exactly as before.

## Error handling

- `begin_output`/`end_output` are pure state flips; they can't fail. If the
  reply leg raises (TTS/timeout), `end_output()` still runs in teardown, so the
  mic is never left stuck muted after an error.
- File-driven runs (`NullPlayer`) and a future non-duplex player are unaffected
  (no-op hooks, `pending()==0`).

## Testing

`DuplexAudio` already injects `stream_factory`/`rate_query`/`poll_timeout` for
hermetic tests; no real device needed. The cascade is tested through
`CascadeDeps` fakes.

Audio-layer (`test` for `DuplexAudio`):

1. **Muted while output active.** `set_output_active(True)`; push a captured
   chunk through the injected callback; assert `capture_frames` yields zeroed
   PCM. (Kills a mutant dropping the `_output_active` term.)
2. **Muted while buffer draining.** `feed()` some audio so `len(_out)>0` with
   `_output_active` clear; assert capture is silenced until the buffer empties,
   then real audio resumes. (Kills a mutant dropping the `len(_out)>0` term.)
3. **User mute composes.** `set_listening(off)` alone still mutes; clearing
   output-active does not un-mute a user-muted mic. (Kills a mutant that ORs the
   gates wrong, e.g. replaces `or` with `and`.)
4. **Open by default.** With everything clear and buffer empty, capture yields
   the real (resampled) audio. (Guards against an always-muted regression.)

Engine-layer (`test_agent_cascade_engine.py`, fake player records calls):

5. **Greeting brackets output.** Assert `begin_output` is called before the
   greeting synth and `end_output` after.
6. **Reply brackets output; thinking does not.** A turn with a `ToolNotice`
   then `SpeechDelta`s: assert `begin_output` fires only when the first clause
   speaks (not during the tool/thinking phase) and `end_output` fires in
   teardown.
7. **`end_output` on every exit.** Barge-in, TTS failure, and timeout paths each
   still call `end_output` (mic never stuck muted). (Kills a mutant that puts
   `end_output` on only the happy path.)

Per the repo gate: each changed line needs an assertion that *fails* if the line
breaks (mutation gate), and the diff needs 100% patch coverage.

## Risks / open questions

- **Sub-100 ms acoustic tail.** Gating on buffer-non-empty covers the digital
  buffer but not the speaker's physical decay / room reverb in the ~one-blocksize
  window right after `_out` empties. A captured chunk straddling that boundary
  could carry faint echo. In practice STT won't form a turn from a sub-100 ms
  fragment, so v1 omits a timed tail to stay clock-free; if field testing shows
  boundary self-interrupts, add a short injectable tail (OpenClaw-style
  watchdog) as a follow-up.
- **Headphone users lose talk-over barge-in unnecessarily.** They have no echo,
  so full-duplex would be safe for them. v1 is uniformly half-duplex; a
  `--full-duplex` opt-out is the natural follow-up (deferred per the config
  decision).
- **Warning copy** is a wording decision to settle during implementation; it
  must stay terse and period-less only if it's option/summary help (this is a
  runtime notice, so normal punctuation is fine).

## Out of scope

- Acoustic echo cancellation / OS audio-processing APIs.
- A `--full-duplex` / `--no-echo-guard` toggle.
- Text-match echo heuristics to preserve talk-over barge-in.
- The spoken tool-call filler (separate spec) — note the filler is audible
  output too, so it is correctly bracketed by `begin_output`/`end_output` once
  both ship.
