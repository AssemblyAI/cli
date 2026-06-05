# Full SDK option coverage for `transcribe` and `stream`

**Date:** 2026-06-03
**Status:** Approved design, ready for implementation plan

## Problem

The CLI currently exposes a tiny slice of the AssemblyAI SDK. `transcribe` offers
only `--speaker-labels` and `--prompt`; `stream` offers `--sample-rate`, `--device`,
and `--prompt`. Meanwhile the SDK's `TranscriptionConfig` exposes ~60 options (PII
redaction, content safety, topic detection, sentiment, auto-chapters, entity
detection, summarization, language detection, custom spelling, word boost, keyterms,
multichannel, audio slicing, webhooks, speaker options, Speech Understanding) and
`StreamingParameters` exposes ~25 (turn detection, voice focus, PII redaction,
speaker labels, encoding, domain, webhooks). None of this is reachable from the CLI.

The goal: make every SDK option controllable from the CLI, without hand-writing 85+
flags or sacrificing discoverability.

## Scope

- **In scope:** full `TranscriptionConfig` coverage on `transcribe`; full
  `StreamingParameters` coverage on `stream`.
- **Out of scope:** LeMUR (deprecated). Speech Understanding speaker-ID-by-name/role
  and custom-formatting get `--config`-only access in v1 (nested shapes, rarely used
  from a shell); translation is promoted to a flag.

## Approach: hybrid (curated flags + escape hatch)

Curated, typed flags cover the common features. A generic escape hatch
(`--config KEY=VALUE`, repeatable, and `--config-file FILE`) covers the long tail.
Both paths share one builder so validation and merge logic live in one place.

### Configuration layering

A single new module, `assemblyai_cli/config_builder.py`, merges three layers and
returns a ready SDK config object. Precedence, lowest to highest:

```
--config-file   (base: full JSON object mapping 1:1 to the SDK config)
      ↓ overlaid by
--config KEY=VALUE   (repeatable ad-hoc overrides)
      ↓ overlaid by
explicit typed flags   (most specific → win)
```

Rationale: an explicit named flag is the most intentional signal, so it wins; a
`--config-file` acts as a reusable profile that flags and `--config` can override.

- **Config file format:** JSON only. Maps 1:1 to the SDK config object's field names.
  Matches the `--json` output convention; one parser.
- **`--config` coercion:** `key=value` strings coerced by the target field's type —
  bool (`true`/`false`/`1`/`0`), int, float, comma-separated list, or JSON for
  complex/nested values.
- **Validation:** unknown keys raise `UsageError` listing valid field names. Enum
  values (speech model, PII policy, summary type/model, encoding, etc.) are validated
  against the SDK enums; invalid values raise `UsageError` listing allowed values.

`config_builder` exposes:

- `KNOWN_TRANSCRIBE_FIELDS` / `KNOWN_STREAM_FIELDS` — field name → coercion type,
  derived from / validated against the SDK config classes.
- `parse_config_overrides(pairs) -> dict` — coerce `key=value` pairs.
- `load_config_file(path) -> dict` — parse and validate a JSON config file.
- `build_transcription_config(flag_values, overrides, file_data) -> aai.TranscriptionConfig`
- `build_streaming_params(flag_values, overrides, file_data) -> StreamingParameters`

## `transcribe` — curated flags

Everything below gets a typed flag. All other `TranscriptionConfig` fields
(`language_confidence_threshold`, `boost_param`, `remove_audio_tags`,
`speaker_options` internals, Speech Understanding speaker-ID / custom-formatting,
etc.) are reachable via `--config` / `--config-file`.

**Model & language**
- `--speech-model {best,nano,slam-1,universal}`
- `--language-code TEXT`
- `--language-detection`
- `--keyterms-prompt TEXT` (repeatable)
- `--temperature FLOAT`

**Formatting**
- `--punctuate / --no-punctuate`
- `--format-text / --no-format-text`
- `--disfluencies`

**Speakers & channels**
- `--speaker-labels` *(exists)*
- `--speakers-expected INT`
- `--multichannel`

**Guardrails**
- `--redact-pii`
- `--redact-pii-policy TEXT` (csv / repeatable)
- `--redact-pii-sub {hash,entity_name}`
- `--redact-pii-audio`
- `--filter-profanity`
- `--content-safety`
- `--content-safety-confidence INT` (25–100)
- `--speech-threshold FLOAT`

**Analysis (auto-rendered)**
- `--summarization`, `--summary-model`, `--summary-type`
- `--auto-chapters`
- `--sentiment-analysis`
- `--entity-detection`
- `--auto-highlights`
- `--topic-detection` (→ `iab_categories`)

**Customization**
- `--word-boost TEXT` (repeatable)
- `--custom-spelling-file FILE` (JSON: `{from: [..], to: ".."}` map)
- `--audio-start INT`, `--audio-end INT` (milliseconds)

**Webhooks**
- `--webhook-url TEXT`
- `--webhook-auth-header NAME:VALUE`

**Speech Understanding**
- `--translate-to TEXT` (repeatable; → `speech_understanding` translation request)

**Escape hatch**
- `--config KEY=VALUE` (repeatable)
- `--config-file FILE`

**Existing, unchanged:** `--llm-gateway-prompt`, `--model`, `--max-tokens`,
`--json`, `--sample`, `--prompt`.

## `stream` — curated flags

**Model & input**
- `--speech-model TEXT`
- `--encoding {pcm_s16le,pcm_mulaw}`
- `--sample-rate INT` *(exists)*
- `--device INT` *(exists)*
- `--language-detection`
- `--domain medical`

**Turn detection**
- `--end-of-turn-confidence-threshold FLOAT`
- `--min-turn-silence INT`
- `--max-turn-silence INT`
- `--vad-threshold FLOAT`
- `--format-turns / --no-format-turns`
- `--include-partial-turns`

**Features**
- `--keyterms-prompt TEXT` (repeatable)
- `--filter-profanity`
- `--speaker-labels`
- `--max-speakers INT`
- `--voice-focus {near_field,far_field}`
- `--voice-focus-threshold FLOAT`
- `--redact-pii`
- `--redact-pii-policy TEXT` (csv / repeatable)
- `--redact-pii-sub {hash,entity_name}`
- `--inactivity-timeout INT`
- `--webhook-url TEXT`
- `--webhook-auth-header NAME:VALUE`

**Escape hatch**
- `--config KEY=VALUE` (repeatable)
- `--config-file FILE`

**Existing, unchanged:** `--prompt`, `--llm-gateway-prompt`, `--model`,
`--max-tokens`, `--json`, `--sample`.

## Result rendering (human mode)

New module `assemblyai_cli/transcribe_render.py`. After printing the transcript text,
it conditionally renders one section per result field present on the returned
transcript object. Each section is a small standalone helper that no-ops when its
field is absent:

- `Summary:` — the summary text/bullets
- `Chapters:` — `start–end  headline` list with formatted timestamps
- `Highlights:` — ranked key phrases
- `Sentiment:` — aggregate percentages plus per-utterance breakdown
- `Entities:` — entity type → text
- `Topics:` — IAB categories with relevance
- `Content Safety:` — flagged labels with confidence/severity

`--json` continues to dump the full raw transcript object, untouched.

## Component touch list

**New**
- `assemblyai_cli/config_builder.py`
- `assemblyai_cli/transcribe_render.py`

**Changed**
- `assemblyai_cli/commands/transcribe.py` — new flags, delegate config to builder
- `assemblyai_cli/commands/stream.py` — new flags, delegate config to builder
- `assemblyai_cli/client.py` — `transcribe()` / `stream_audio()` accept prebuilt
  config objects; `transcribe()` returns the full transcript with all result fields
- `assemblyai_cli/render.py` — shared rendering helpers (timestamp formatting, etc.)
- `README.md` — document the new options and escape hatch
- `assemblyai_cli/templates/` — refresh sample scripts to show a few options

## Testing (exhaustive)

- **Unit — `config_builder`:** type coercion per kind (bool/int/float/list/json);
  layer precedence (file < `--config` < flags); unknown-key error; enum-validation
  errors; property tests for coercion round-trips; a parametrized case per curated
  flag asserting it lands on the right SDK field.
- **Unit — `transcribe_render`:** one test per section using fake transcript
  fixtures, including the absent-field no-op path.
- **Command:** `transcribe` / `stream` build the expected config object from flags
  (mocked client), and the escape hatch merges correctly.
- **e2e:** real-API runs covering several analysis features (e.g. summarization +
  chapters + sentiment), following the repo's existing e2e pattern.

## Build sequence

1. `config_builder` + its tests (foundation, no UI).
2. `transcribe`: flags → builder → client → `transcribe_render` + tests.
3. `stream`: flags → builder → client + tests.
4. README + samples.
5. e2e additions.
