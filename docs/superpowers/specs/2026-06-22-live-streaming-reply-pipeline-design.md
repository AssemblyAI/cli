# `assembly live` — streaming reply pipeline (lower time-to-first-audio)

**Date:** 2026-06-22
**Status:** Approved, ready for implementation plan
**Area:** `aai_cli/agent_cascade/` (`engine.py`, `brain.py`, `text.py`), `aai_cli/tts/session.py` (consumed, not changed)

## Problem

Today a live cascade turn runs in series: the whole deepagents graph is driven to
completion (`engine._complete_within`), the finished reply is split into sentences
(`text.split_sentences`), and each sentence is synthesized with the **buffered** TTS
path (`tts_session.synthesize(...).pcm`) before any audio plays. So time-to-first-audio
is `full-LLM-generation + first-sentence-synthesis`, with no overlap.

The system prompt caps replies at 1–2 sentences, so sentence-level pipelining alone
would not help the common single-sentence reply (you still can't speak a sentence until
it is fully generated). The win for the typical short reply requires overlapping work
*within* an utterance.

## Goal

Overlap the three stages — token generation, synthesis, and playback — so audio starts
as soon as the first clause is ready, even for a single-sentence reply. Approach chosen
(over sentence-only and token+sentence variants): **token streaming + clause-level
flush + streaming TTS frames**.

Non-goals: changing the model, the 1–2 sentence prompt guidance, STT/`format_turns`
behavior, or the front-end protocols (`Renderer`/`Player`).

## Design

### 1. Brain: a reply event stream (`brain.py`)

Replace `build_completer` (returns `str`) with a streaming producer:

```
build_streamer(api_key, config, *, graph=None) -> Callable[[messages], Iterator[ReplyEvent]]
```

`stream_reply(messages)` drops the prepended `system` message (as today), then iterates
`graph.stream(input, stream_mode="messages")` and yields two frozen event dataclasses
defined in `brain.py`:

- `SpeechDelta(text: str)` — a top-level assistant-text token delta. Only
  `AIMessageChunk.content` deltas are yielded. Subagent tokens are excluded automatically
  because we do **not** pass `subgraphs=True`; tool-call AIMessage chunks carry no spoken
  content and so contribute nothing here.
- `ToolNotice(label: str)` — emitted when a tool-call chunk lands, carrying the speakable
  label from `_tool_label` (e.g. "Searching the web").

Graph failures wrap into `CLIError` exactly as `_run_graph` does today, raised out of the
iterator (the consumer surfaces it). Verbose `-v` flow logging (`_FLOW_LOG`) moves inside
this same streaming loop — logging tool calls/results/interim assistant text as chunks
arrive (strictly better than today's `stream_mode="values"` snapshot logging).

`complete_reply`, `_complete_within`, `_run_graph`/`_drive_graph`'s invoke branch, and
`_reply_text` are removed; the always-stream path supersedes them.

### 2. TTS: a frame sink instead of buffered bytes

`CascadeDeps.synthesize` changes from `Callable[[str], bytes]` to
`Callable[[str, Callable[[bytes], None]], None]`, implemented over the existing streaming
primitive `tts_session.synthesize(api_key, spec, on_audio=...)` (`tts/session.py:234`,
already used by `assembly speak`). The engine's sink is `_feed(pcm)`, which enqueues to
the player **only when `_stop` is not set** — a barge-in therefore just drops the
remaining frames of the in-flight clause; no exception is threaded through the TTS module.
The greeting uses the same sink.

### 3. Engine: streaming `_generate_reply` (`engine.py`)

The graph stream runs on a **throwaway daemon producer thread** that pushes typed items
onto a `queue.Queue`; the reply worker thread consumes them. The producer thread preserves
today's wall-clock backstop: a stalled gateway can block inside a token read that the
worker cannot otherwise observe — the same reason `_complete_within` used a throwaway
thread. The consumer's `queue.get` uses a `time.monotonic` deadline so the total-turn
timeout and its "took longer than {n}s to respond" message are unchanged
(`_REPLY_TIMEOUT_SECONDS` stays 60s). On timeout the producer is abandoned (daemon, dies
with the process) and a `CLIError(error_type="agent_timeout")` is raised, as today.

Producer items: `ToolNotice`/`SpeechDelta` (forwarded from the brain), plus engine
sentinels `Done` and `Error(exc)`.

Consumer loop, per item:

- `ToolNotice(label)` → `renderer.tool_call(label)` **and clear the pending clause
  buffer** (the "drop unspoken preamble on a tool call" decision). Rendering lives on the
  consumer thread, so the buffer clear is same-thread.
- `SpeechDelta(text)` → on the first delta of the turn, `_speaking.set()` then
  `renderer.reply_started()`; append `text` to the buffer; flush any complete clauses to
  TTS via `pop_clauses`, checking `_stop` between clauses.
- `Done` → flush the buffered tail as a final clause; join spoken clauses and append to
  history (then `trim_history`); `_speaking.clear()`; `renderer.reply_done(interrupted=
  self._stop.is_set())`.
- `Error(exc)` → if nothing has been spoken yet, the existing pre-speak path
  (`reply_started` + `(error: {message})` transcript + `reply_done`); otherwise
  `_record_error` and stop.

`_speaking` is set only once the turn begins speaking (first `SpeechDelta`), preserving
the "Ctrl-C quits while thinking, interrupts while speaking" semantics in `_silence` /
`interrupt_reply`. Barge-in (`_barge_in`), the interrupt path, and the sliding-history
window are otherwise unchanged.

### 4. Incremental clause splitter (`text.py`)

Add a pure function:

```
pop_clauses(buffer: str, *, min_chars: int) -> tuple[list[str], str]
```

- **Hard boundaries** `.!?` flush a clause when the terminator is followed by whitespace
  (reusing `split_sentences`' rule, so `$3.50` / `...` don't fragment).
- **Soft boundaries** `,;:` (followed by whitespace) flush only when the pending clause is
  at least `min_chars` long, avoiding choppy two-word TTS fragments.
- The text after the last boundary is returned as `remainder` and kept buffered by the
  engine; the stream-end tail is flushed on `Done`.

`min_chars` is a module constant (~25), marked `# pragma: no mutate` (a ±1 shift is
behaviorally equivalent). `pop_clauses` is pure and table-tested.

### Data flow

```
STT final turn
  -> producer thread: graph.stream(messages) -> queue[ToolNotice|SpeechDelta|Done|Error]
  -> reply worker: queue.get(deadline)
       ToolNotice -> renderer.tool_call + clear buffer
       SpeechDelta -> buffer += text; pop_clauses -> for clause: synthesize(clause, _feed)
       _feed(pcm) -> player.enqueue (skipped once _stop set)
       Done -> flush tail, record history, reply_done
```

## Error handling

- LLM/graph/tool failure → `CLIError` from the iterator → `Error` item → pre-speak or
  mid-speak handling above; first failure recorded in `session.error` and re-raised on the
  main thread by `run_cascade` (unchanged).
- TTS failure during a clause → `CLIError` from `synthesize` → `_record_error` + stop
  (mirrors today's per-sentence synth failure).
- Total-turn stall → `monotonic` deadline on `queue.get` → `agent_timeout` `CLIError`.

## Testing

- **Engine** (`tests/test_agent_cascade_engine.py`): inject a fake `stream_reply` yielding
  scripted `SpeechDelta`/`ToolNotice`/raising `CLIError`, and a fake `synthesize` recording
  its sink calls. Assert: clause boundaries trigger synth at the right points; a
  `ToolNotice` clears the unspoken buffer; barge-in mid-stream stops further enqueue and
  records only spoken text; the `monotonic` deadline raises the timeout error; the
  pre-speak and mid-speak error paths render correctly. No graph, socket, mic, or speaker.
- **Brain** (`tests/test_agent_cascade_*`): inject a fake `graph` whose `.stream` yields
  `(chunk, metadata)` tuples; assert top-level text deltas become `SpeechDelta`, tool-call
  chunks become `ToolNotice`, subagent/tool chunks are filtered, and graph exceptions wrap
  to `CLIError`. Verbose logging asserted via the `_FLOW_LOG` records.
- **`pop_clauses`** (`tests/test_agent_cascade_*` or the text-helper test): table tests for
  hard/soft boundaries, the `min_chars` guard, `$3.50`/`...` non-fragmentation, and tail
  handling.

Coverage/mutation gates: the new branches (clause flush conditions, the buffer-clear, the
deadline expiry, `_speaking` first-delta gate) each need an assertion that fails if the
line breaks, not just coverage.

## Risks / mitigations

- **Gateway token streaming reliability** — confirmed: `assembly code` streams through the
  gateway, and the streaming tool-call-id bug was fixed in `model.py` (PR #247).
- **Barge-in responsiveness** — improves vs today: frame-level enqueue drop + clause-level
  `_stop` checks replace whole-sentence granularity.
- **Choppy TTS from over-eager flushing** — guarded by the `min_chars` soft-boundary
  threshold.

## Out of scope (possible follow-ups)

- `--no-format-turns` fast mode (shaves the STT formatting round-trip).
- Routing the no-tools case to a plain completion instead of the full deepagents graph
  (reduces per-request token overhead / time-to-first-token).
