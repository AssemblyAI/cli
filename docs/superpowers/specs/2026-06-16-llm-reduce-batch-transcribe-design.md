# `--llm-reduce`: map-reduce LLM step for batch transcribe

Date: 2026-06-16

## Problem

`assembly transcribe --from-stdin --llm 'PROMPT'` runs an LLM prompt over *each*
transcript (a per-source "map") and writes the result into each source's sidecar
under the `transform` key. There is no built-in way to run a single LLM prompt
over *all* of a batch's results to produce one aggregate answer.

Piping the batch into `assembly llm` does not achieve this: batch mode writes a
Rich status table to stdout (human mode) or per-source `{type:"result"}` NDJSON
records (`--json` mode) — neither contains the transcript text or the per-source
`--llm` output, which live only in the sidecar files. So
`assembly transcribe --from-stdin --llm '...' | assembly llm 'summarize'` would
summarize a status table, not the results.

## Goal

Add a repeatable `--llm-reduce 'PROMPT'` flag to `assembly transcribe` that runs
one LLM Gateway call (a "reduce") over the combined per-source results and prints
the aggregate answer to stdout — making the command itself the cleanly pipeable
unit, with no second `assembly llm` needed.

## Design

### Concept: map-reduce

- `--llm` (existing) = the per-source **map**: runs over each transcript.
- `--llm-reduce` (new) = the **reduce**: one chain over all sources' results.

`--llm-reduce` is repeatable (a prompt chain, mirroring `--llm`) and reuses the
same `--model` / `--max-tokens` already carried for the map step.

### Reduce input

For each **completed or skipped** source, the reduce input is:

- the **last `--llm` map step's output** if a map chain ran, otherwise
- the source's **transcript text**.

Items are concatenated with a per-source header (`### Source: <source>`) so the
reduce prompt can attribute findings. Failed sources are excluded.

The combined text is sent inline via the existing `llm.run_chain(...,
transcript_text=combined)` path — the same inlining `stream --llm` already uses
(there is no single transcript id to inject server-side for an aggregate).

### Components / data flow

1. **`commands/transcribe.py`** — add a repeatable `--llm-reduce` Typer option in
   the existing LLM help panel (terse, period-less help). Parse into the existing
   `TransformOptions` via a new `reduce_prompts: tuple[str, ...]` field.
2. **`app/transcribe/batch.py`** — after `_drain`/`_summarize`, when
   `reduce_prompts` is non-empty: gather reduce inputs from the sidecars the batch
   already wrote, build the combined text, call `llm.run_chain(...)` once, and
   print the result to **stdout** (`output.console`).
3. **Output routing** — when `--llm-reduce` is set, render the batch progress
   table to **stderr** (`error_console`) instead of stdout, so stdout carries only
   the reduce result and the command pipes cleanly. Gated on the flag: with no
   `--llm-reduce`, existing batch output (and its snapshots) is unchanged.
4. **`--json` mode** — keep the per-source `{type:"result"}` NDJSON, then emit a
   final additive `{type:"reduce", model, prompts, output}` record. Additive to
   the documented "every NDJSON line carries a `type`" contract.
5. **Single source (non-batch)** — there is nothing to aggregate, so
   `reduce_prompts` are appended to the `--llm` chain over the one transcript in
   `app/transcribe/run.py` (effectively `prompts + reduce_prompts` run as one
   chain). No error, predictable behavior.

### End state (one command, not a pipe)

```sh
assembly transcribe --from-stdin --concurrency 3 --speaker-labels \
  --llm 'Judge diarization quality; output JSON {speaker_count, issues, score}' \
  --llm-reduce 'Rank these videos worst-to-best and summarize the failure modes'
```

## Testing (to clear the gates)

- Batch reduce: assert the reduce LLM call receives the concatenated per-source
  map outputs (with headers) and the result is written to stdout.
- Reduce-input fallback: a source with no `--llm` map contributes its transcript
  text.
- Single source: `--llm-reduce` appends to the `--llm` chain over the one
  transcript (no aggregation path).
- `--json`: a final `{type:"reduce"}` record is emitted after the per-source
  `{type:"result"}` records.
- Routing: the progress table goes to stderr only when `--llm-reduce` is set;
  unchanged otherwise.
- Regenerate the `transcribe --help` snapshot; add a REFERENCE.md / README entry
  (docs-consistency gate).

Tests construct `TransformOptions` / run-path data directly (per the repo's
options/run seam) and assert behavior, so the mutation gate's changed lines are
killed by failing-on-break assertions, not mere coverage.

## Scope / non-goals

- Touches only `transcribe`'s own modules (`commands/transcribe.py`,
  `app/transcribe/*`) plus docs/snapshots — no shared-file edits, consistent with
  the additive-command convention.
- No change to the existing `--llm` map semantics or to batch output when
  `--llm-reduce` is absent.
- Dependency set unchanged (`uv.lock` untouched).
