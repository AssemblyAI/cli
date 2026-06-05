# CLI Color Theme — Design

**Date:** 2026-06-03
**Status:** Approved

## Goal

Give `aai` a single, idiomatic Rich color theme and apply it consistently to all
human-facing output. Today color is ad-hoc: inline `[red]` / `[yellow]` /
`[green]` / `[dim]` markup scattered across a few command files, while the
streaming and Voice Agent renderers print fully uncolored `Text`. There is no
central palette.

Direction chosen with the user: **brand accent + semantic palette**, applied at
**full scope** (role/speaker labels, lifecycle notices, table status values,
errors, and notices). Transcript body text stays default for readability.

## Non-goals

- No new user-facing flags (no `--no-color`/`--color`). Rich already honors
  `NO_COLOR` and disables ANSI on non-TTYs; the agentic/CI path routes to JSON.
- No restyling of JSON output. JSON stays plain, pipe-safe NDJSON / `json.dumps`.
- No unrelated refactoring of the command modules.

## Architecture

### New module: `assemblyai_cli/theme.py`

Single source of truth. Exports:

- `BRAND = "#2545D3"` — AssemblyAI brand accent, defined once so it can be
  swapped in one place.
- `THEME: rich.theme.Theme` mapping **semantic style names** so call sites never
  hard-code raw colors again:
  - `aai.brand` → `bold #2545D3`
  - `aai.heading` → `bold #2545D3`
  - `aai.label` → `#2545D3` (role/speaker label prefixes)
  - `aai.success` → `green`
  - `aai.error` → `bold red`
  - `aai.warn` → `yellow`
  - `aai.muted` → `dim`
  - `aai.speaker.0` … `aai.speaker.N` → a small rotating palette
    (e.g. brand blue, cyan, magenta, green, yellow) for distinct, deterministic
    per-speaker label colors.
- `SPEAKER_STYLES: tuple[str, ...]` — the speaker style names in rotation order.
- `make_console(file=None) -> rich.console.Console` — builds every `Console`
  **with `theme=THEME` attached**, so style names resolve globally.
- `speaker_style(speaker) -> str` — deterministic map from a speaker id
  (e.g. `"A"`, `"B"`, `0`, `1`) to one of `SPEAKER_STYLES`.
- `status_style(status: str) -> str` — map a transcript/step status string to a
  style name: `completed`/`installed`/`removed`/`ok` → `aai.success`;
  `error`/`failed` → `aai.error`; `queued`/`processing`/in-progress → `aai.warn`;
  otherwise `aai.muted`.

### Routing all consoles through the theme

- `output.console` is created via `theme.make_console()`.
- `BaseRenderer._console_obj()` creates its per-stream console via
  `theme.make_console(file=self.out)` instead of bare `Console(file=...)`.

This is the only structural change; everything else is markup/style edits.

## Component changes

- **`output.py`**
  - `console = theme.make_console()`.
  - `emit_error`: `[red]Error:[/red]` → `[aai.error]Error:[/aai.error]`.

- **`render.py` (`BaseRenderer`)**
  - Line helpers (`_update_line`, `_finalize_line`, `_line`) accept `str | Text`
    instead of only `str`. When given a `str` they wrap in `Text(text)` as today
    (no markup parsing — preserves current behavior); when given a `Text` they
    use it directly. `stopped()` renders "Stopped." in `aai.muted`.

- **`streaming/render.py` (`StreamRenderer`)**
  - `begin`: "Listening… (Ctrl-C to stop)" in `aai.muted`.
  - `turn`: body text stays default.
  - `llm`: `💡 …` line in `aai.brand`.

- **`agent/render.py` (`AgentRenderer`)**
  - `connected`: notice in `aai.muted`.
  - `user_partial`/`user_final`: `you:` label in `aai.label`, body default
    (build a `Text` with a styled label span).
  - `agent_transcript`: `agent:` label in `aai.label`, body default.

- **`commands/transcribe.py`**
  - `_render_transcript`: each `Speaker X:` label styled via
    `theme.speaker_style(u["speaker"])`; body default. Returns a `Text` (or
    themed markup) rather than a plain escaped string. Plain (non-diarized)
    text path is unchanged.

- **`commands/transcripts.py`**
  - `list` table: header styled `aai.heading`; the `status` cell colored via
    `theme.status_style(...)`.

- **`commands/llm.py`** — output body unchanged (plain model text), but uses the
  themed console for free; no markup added.

- **`commands/login.py`** — `[green]Authenticated[/green]` → `[aai.success]`;
  `[dim]…[/dim]` → `[aai.muted]`.

- **`commands/samples.py`** — `[yellow]Note:[/yellow]` → `[aai.warn]`.

- **`commands/claude.py`** — `_render_steps`: each step's status colored via
  `theme.status_style(s["status"])`; heading in `aai.heading`.

## Data flow

Command body → `output.emit(data, renderer, json_mode)`:
- JSON mode → unchanged plain JSON.
- Human mode → `console.print(renderer(data))`, where `console` carries `THEME`,
  so any `[aai.*]` markup or `style="aai.*"` `Text` resolves to the palette.
  Rich strips ANSI automatically when the console's file is not a TTY.

## Error handling

No new failure modes. Unknown status strings fall through `status_style` to
`aai.muted`. `speaker_style` is total over any hashable speaker id via modulo
over `SPEAKER_STYLES`.

## Testing (TDD — tests first)

New `tests/test_theme.py`:
- `THEME` resolves each named style without raising.
- `status_style` maps representative statuses to the right style names,
  including the unknown-status → `aai.muted` fallback.
- `speaker_style` is deterministic and stays within `SPEAKER_STYLES`.

Renderer/command tests use a **forced-terminal** themed console
(`make_console` + `force_terminal=True`) and assert that styled output contains
the expected ANSI/markup for labels and statuses; existing plain-capture tests
(non-TTY) keep passing because Rich emits no ANSI there. Run the full existing
suite to confirm no regressions in the many `tests/test_*` assertions.

## Files

- New: `assemblyai_cli/theme.py`, `tests/test_theme.py`
- Edit: `output.py`, `render.py`, `streaming/render.py`, `agent/render.py`,
  `commands/{transcribe,transcripts,llm,login,samples,claude}.py`,
  and the corresponding `tests/test_*` files.
