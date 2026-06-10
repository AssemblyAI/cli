# `aai speak` — speaker-aware multi-voice playback

**Date:** 2026-06-10
**Status:** Approved design

## Summary

Extend `aai speak` so that when its input is speaker-labeled transcript text —
the output of `aai transcribe … --speaker-labels` — it strips the `Speaker X:`
labels and synthesizes each speaker in a different voice, concatenated into one
stream. The motivating pipe should "just work":

```sh
aai transcribe "https://youtu.be/…" --speaker-labels | aai speak --sandbox
```

Plain (unlabeled) text keeps today's behavior exactly: one voice, spoken
literally. Detection is automatic — no new flag is required to opt in.

## Behavior

### Detection (automatic)

After reading the text (argument or stdin, unchanged), inspect it: if the
**first non-blank line matches `^Speaker <id>:`** followed by a space (id = one
or more non-space characters), the input is treated as speaker-labeled and enters multi-voice
mode. Otherwise `speak` behaves as it does today (single voice, literal text).
Anchoring on the first labeled line means ordinary prose never misfires into
multi-voice mode.

### Parsing (folds the 80-column wrap)

`aai transcribe --speaker-labels` renders one utterance per logical line, but its
Rich console soft-wraps at ~80 columns when piped, so a long utterance becomes
several physical lines and only the first carries the `Speaker X:` prefix. The
parser accounts for this:

- A line matching `^Speaker (?P<id>\S+): (?P<text>.*)$` **starts a new turn**.
- Any following line that does **not** match that pattern is a wrapped
  continuation and is appended to the current turn's text (joined with a single
  space; surrounding whitespace stripped).
- Blank lines are skipped.
- **Consecutive turns by the same speaker are merged** into one synthesis
  segment (fewer connections, more natural delivery).

The result is an ordered list of `Segment(speaker_id, text)` with labels
removed. A label line with empty text contributes no text but still establishes
the speaker for following continuation lines.

### Voice assignment

- **Default rotation:** `["jane", "michael", "mary", "paul", "eve", "george"]`
  (all confirmed-working PocketTTS English voices, ordered to alternate timbre).
  Speakers are assigned in **first-appearance order**, wrapping when there are
  more speakers than rotation entries.
- **Per-speaker override:** `--voice` is repeatable and accepts a
  `SPEAKER=VOICE` form, e.g. `--voice A=vera --voice B=paul`. An explicit mapping
  wins over the rotation for that speaker. The speaker id is matched
  case-insensitively against the parsed ids.
- **Bare `--voice NAME`** (no `=`) keeps its current meaning: the voice for
  single-voice (unlabeled) mode (falling back to `jane` when none is given). In
  multi-voice mode an *explicitly passed* bare `--voice` is **ignored with a
  one-line stderr note** pointing at the `--voice A=NAME` form, so it never
  silently collapses every speaker to one voice. The note fires only when the
  user actually typed a bare voice — see the empty-list default below.
- An override naming an unknown/erroring voice is not validated client-side; it
  surfaces as the normal TTS error at synthesis time.

### Synthesis & output

- Each segment is synthesized with its assigned voice via `session.synthesize`
  (one WebSocket connection per segment — voice is fixed at connect time, so a
  voice change needs a new connection; merging consecutive same-speaker turns
  keeps this minimal). `--language` and `--sample-rate` apply to every segment.
- Segments are concatenated into one 16-bit mono PCM buffer with **~250 ms of
  silence between turns** for natural pacing (no leading/trailing pad).
- Output is unchanged in spirit: play through the speakers (interruptible —
  Ctrl-C aborts immediately) or write one concatenated WAV with `--out`.

### JSON output

- **Multi-voice mode:** one object —
  `{"mode": "multi", "speakers": {"A": "jane", "B": "michael"}, "segments": N,
  "sample_rate": int, "audio_duration_seconds": float, "bytes": int,
  "out": str|null}`. `speakers` lists the resolved id→voice map in
  first-appearance order.
- **Single-voice mode:** the existing shape is unchanged
  (`{voice, language, sample_rate, audio_duration_seconds, bytes, out}`).

## Components

- **`aai_cli/tts/dialogue.py`** (new) — pure, I/O-free, unit-testable:
  - `looks_like_speaker_labeled(text: str) -> bool`
  - `parse_segments(text: str) -> list[Segment]` (`Segment` = frozen
    `(speaker_id, text)` dataclass)
  - `parse_voice_overrides(values: list[str]) -> tuple[str | None, dict[str, str]]`
    — splits repeatable `--voice` values into the bare voice (if any) and the
    `SPEAKER→VOICE` map.
  - `assign_voices(segments, rotation, overrides) -> list[tuple[str, str]]`
    — `(voice, text)` per segment, applying overrides then first-appearance
    rotation. Also returns/exposes the id→voice map for JSON.
  - `DEFAULT_VOICE_ROTATION` constant.

- **`aai_cli/tts/session.py`** — add
  `synthesize_dialogue(api_key, segments, *, language, sample_rate, connect,
  on_warning) -> SpeakResult`, looping `synthesize` per `(voice, text)` segment
  and concatenating PCM with an inter-turn silence. Reuses the existing
  connect/auth-mapping path. `SpeakResult` is unchanged.

- **`aai_cli/tts/audio.py`** — add `silence(sample_rate, seconds) -> bytes`
  (zeroed 16-bit mono PCM) for the inter-turn gap.

- **`aai_cli/commands/speak.py`** — `--voice` becomes repeatable
  (`list[str]`, default **`[]`**); single-voice mode uses the bare value if given
  else `jane`. The empty default lets the command tell "user passed a bare voice"
  from "default", so the multi-mode note fires only on an explicit bare voice.
  Flow: read text → detect → if labeled, parse + resolve voices +
  `synthesize_dialogue`; else single-voice `synthesize` as today → play or write
  → emit human/JSON result. The bare-voice-in-multi-mode note is printed here.

## Error handling

| Condition | Result |
| --- | --- |
| Labeled input but a chosen voice errors mid-stream | the segment's `synthesize` raises the existing clean `APIError`; the command fails (no partial playback) |
| `--voice` value with an empty side (`=jane`, `A=`) | `UsageError` (exit 2) with the expected `SPEAKER=VOICE` form |
| Bare `--voice` in multi-voice mode | honored-but-ignored: one stderr note, synthesis proceeds with rotation/overrides |
| Empty text after stripping (only labels, no content) | existing "No text to speak" usage error (exit 2) |
| Production environment / no key / Ctrl-C | unchanged from current `speak` behavior |

## Testing

Must clear the mutation and 100%-patch-coverage tail gates — assert behavior,
not just execution.

- **`dialogue.py`** (pure functions, the bulk of the logic):
  - detection: labeled first line → true; prose / leading blank lines / a
    `Speaker`-like word mid-sentence → false.
  - parsing: single-line turns; an utterance wrapped across multiple physical
    lines folds back into one segment with the right text; consecutive
    same-speaker turns merge; blank lines skipped; a label line with empty text.
  - override parsing: bare voice captured; `A=vera` mapped; case-insensitive id;
    malformed (`=x`, `a=`) → error.
  - assignment: first-appearance rotation, wrap past the rotation length,
    override beats rotation, single-speaker labeled input → one voice.
- **`session.synthesize_dialogue`**: injected fake socket(s) drive several
  segments; assert the per-segment voice query param, PCM concatenation order,
  and the inserted silence length between turns (and none at the ends).
- **`audio.silence`**: returns exactly `seconds * sample_rate` frames of zero
  bytes (even length).
- **`commands/speak.py`** (Typer `CliRunner`): a labeled stdin pipe →
  multi-voice path with the expected id→voice map; `--voice A=vera` override
  reflected; bare `--voice` in multi mode emits the note and still rotates;
  `--json` multi shape; `--out` writes one WAV; single-voice path unchanged.
- Regenerate the `aai speak --help` syrupy snapshot for the repeatable `--voice`
  option; never hand-edit `.ambr`.

## Out of scope (v1)

- A `--gap`/`--no-gap` flag (fixed ~250 ms pause for now).
- Non-AssemblyAI label formats / custom label regexes.
- Per-speaker language or sample-rate.
- Reusing one WebSocket across consecutive different-voice segments (a new
  connection per voice change is acceptable).
- Client-side validation of voice names against a catalog.
