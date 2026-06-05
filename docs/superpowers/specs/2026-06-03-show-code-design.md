# `--show-code`: graduate from CLI to SDK code

**Date:** 2026-06-03
**Status:** Design — pending implementation

## Purpose

Help a user who is exploring AssemblyAI through the CLI move to writing their own
SDK code. `--show-code` turns any `transcribe`/`stream`/`agent` invocation into the
idiomatic Python that would do the same thing — built from the exact flags you
passed — so you can copy it into your own app.

This is an **onboarding / "graduate to code"** feature. The emitted code is a
teaching artifact: readable, commented, copy-pasteable SDK Python a human would
actually want to start from — not a verbatim dump of internal state.

## Scope

- **Commands:** `transcribe`, `stream`, `agent`.
- **Trigger:** a `--show-code` flag on each. It is a **print-only mode**: the
  command builds its config from the flags, prints the equivalent Python, and
  **exits without running** — no API call, no upload, no microphone, no streaming
  session. (It does not require auth, since the generated code reads the key from
  the environment and nothing is sent.)
- **Output:** the raw Python is written to stdout via the builtin `print` (not the
  Rich console), so it can be redirected straight into a runnable file —
  `aai transcribe foo.mp3 --show-code > my_script.py` — with no header, no ANSI,
  and no `[...]`-as-markup mangling. `--json` has no effect in this mode (the code
  *is* the output). If code generation fails, that is a real error (it is the
  whole job), not a swallowed warning.
- **Language:** Python only.
- **Completeness:** a fully runnable script — imports, auth setup, config, the
  call, and result handling that mirrors the features that were enabled.

### Explicitly out of scope (YAGNI)

- Other languages (TypeScript/Go). The serializer/snippet design leaves room to
  add them later, but we ship Python only.
- Augmenting a real run (printing code *and* executing). An earlier draft did
  this; `--show-code` is now print-only — it never executes the operation.
- Writing the code to a file. `--show-code` prints to stdout. (`samples create`
  already covers the write-to-disk story.)

## Why not unify with the renderers (rejected alternative)

A tempting "zero-drift" approach is a single feature registry that drives *both*
the live Rich terminal rendering and the code generator. Rejected because:

- The two outputs have different jobs — Rich panels/tables vs. plain teaching
  Python — so the shared abstraction would need `if rendering vs generating`
  branches everywhere (drift in disguise).
- It forces refactoring stable, tested renderers to serve a new feature.
- It couples terminal cosmetics to generated code forever.

Instead we keep the generator **separate from** the renderers and use tests to
guarantee they don't drift (see Testing). This trades "prevent drift by
construction" for "detect drift in CI", which is the right trade for a finite,
slow-moving feature list.

## Architecture (Approach B)

Three small, independently testable pieces plus a thin flag wiring.

### 1. Config serializer — `assemblyai_cli/code_gen/serialize.py`

The inverse of `config_builder`. Takes the **actual built config object** the
command already constructed (`TranscriptionConfig` for transcribe,
`StreamingParameters` for stream) and emits Python source for only the
**non-default** fields, as keyword arguments.

- Source of truth: the live config object, so new config fields flow through
  with no generator change.
- Compares each field against the SDK default; emits only differences. This
  keeps generated code short and readable (the whole point).
- Renders SDK types correctly: enums as `SpeechModel.u3_rt_pro`, nested
  structures (e.g. the `speech_understanding` translation dict, custom spelling)
  as literals.
- Returns a list of `"field=value"` lines that the template indents into the
  config constructor.

Round-trip invariant (enforced by test): for any config the CLI can build,
`build_config(eval(serialize(config))) == config`.

### 2. Per-command skeleton templates — reuse `assemblyai_cli/templates/`

The existing `transcribe.py.tmpl` / `stream.py.tmpl` / `agent.py.tmpl` files are
the stable boilerplate (imports, auth, the call). We extend them with two slots:

- A config slot where serialized `field=value` lines are injected.
- A result-handling slot where feature snippets are injected (transcribe only;
  stream/agent result handling is fixed).

**Auth difference from `samples create`:** `samples create` writes to disk
(0600) and injects the literal API key. `--show-code` prints to the terminal and
scrollback, so it must **not** echo the secret. Generated code uses:

```python
aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
```

with a one-line comment telling the user to export their key. (For `agent`, the
`Authorization: Bearer` header reads the same env var.)

### 3. Feature-snippet table — `assemblyai_cli/code_gen/snippets.py`

Transcribe only. A table mapping an enabled analysis feature to the Python that
reproduces its output. It mirrors the existing one-function-per-feature shape of
`transcribe_render.py` (`_render_summary`, `_render_chapters`,
`_render_sentiment`, `_render_entities`, `_render_topics`, `_render_highlights`,
`_render_content_safety`, speaker-label utterances).

Each entry: `predicate(config) -> bool` (was the feature enabled?) and a code
`snippet: str`. The generator appends the snippet for each enabled feature in a
stable order. A feature with no snippet is a visible, tested gap — not silent
drift.

### 4. Flag wiring

Each command builds its config object today (e.g. `transcribe.py:155`
`tc = config_builder.build_transcription_config(...)`). After the real call, if
`--show-code` is set and not in `--json` mode, call
`code_gen.render(command, config_object, source)` and print the result below the
normal output, in a visually distinct block.

## Data flow (transcribe example)

```
flags --> config_builder.build_transcription_config() --> tc (TranscriptionConfig)
                                                            |
                          real call: client.transcribe() <--+
                                                            |
                                  --show-code? --> code_gen.render("transcribe", tc, source)
                                                            |
                          serialize(tc) -> config lines     |
                          snippets(tc)  -> feature blocks ---+--> template --> printed Python
```

## Error handling

- Code generation runs **after** the successful API call, so a generation bug
  never blocks the user's actual result.
- Generation is wrapped so that a failure prints a short warning
  (`could not render sample code: …`) rather than crashing the command. The
  transcript/stream/agent output is the contract; the code is a bonus.
- Suppressed entirely in `--json` mode (machine-readable output stays clean).

## Testing — invalid code must be impossible to generate without a failure

The requirement is strict: it must be impossible to produce invalid code without
a test failing. We get there structurally, not by enumerating examples. Two facts
make it tractable: (a) generation is driven *entirely* by the merged-kwargs dict,
and (b) `config_builder.TRANSCRIBE_COERCE` / `STREAM_COERCE` are the authoritative
valid-field sets. So a hypothesis strategy built *from those tables* blankets the
entire legal input space, and any field added later is fuzzed automatically.

1. **Fuzz-compiles (syntactic validity):** fuzz the full config domain through
   every renderer and `compile()` every output (`compile` is stricter than
   `ast.parse` and is what `python file.py` runs). No syntactically invalid Python
   can be produced. The agent renderer is additionally fuzzed with arbitrary text
   (quotes, newlines, backslashes, unicode) injected via `repr`.
2. **Round-trip (config fidelity):** for any fuzzed config, the
   `TranscriptionConfig(...)` / `StreamingParameters(...)` the generated code
   builds must `eval` back to the original merged dict. Guarantees the emitted
   call reconstructs the same config; catches any dropped/mangled field.
3. **Result-handling execs:** every snippet is `exec`'d against a stub transcript
   that exposes the attributes the SDK provides. A typo'd attribute in any snippet
   fails here.
4. **Coverage guard:** every analysis feature with a `_render_*` function in
   `transcribe_render.py` must have a snippet entry (or a documented exclusion).
   The tripwire for "added a feature, forgot the snippet."

What tests *cannot* assert: that a snippet's wording is the *clearest* phrasing —
that's editorial judgment, reviewed by a human, not enforced by a test.

## Files

| File | Change |
| --- | --- |
| `assemblyai_cli/code_gen/__init__.py` | New. `render(command, config, source)` entry point. |
| `assemblyai_cli/code_gen/serialize.py` | New. Config object → non-default `field=value` lines. |
| `assemblyai_cli/code_gen/snippets.py` | New. Feature → result-handling snippet table. |
| `assemblyai_cli/templates/*.py.tmpl` | Extend with config + result-handling slots; env-var auth. |
| `assemblyai_cli/commands/transcribe.py` | Add `--show-code`; call `code_gen.render` after the call. |
| `assemblyai_cli/commands/stream.py` | Add `--show-code`; same wiring. |
| `assemblyai_cli/commands/agent.py` | Add `--show-code`; same wiring. |
| `tests/test_code_gen.py` | New. Round-trip, golden, coverage-guard tests. |

## Build sequence

1. `serialize.py` + round-trip property test (no flag wiring yet).
2. Extend `transcribe.py.tmpl`; `snippets.py` + coverage-guard test.
3. Wire `--show-code` into `transcribe`; golden + executes-clean tests.
4. Repeat wiring for `stream` (config serializer reused, fixed result handling).
5. `agent` (no `TranscriptionConfig`; serialize its session params/flags; fixed
   result handling).
```
