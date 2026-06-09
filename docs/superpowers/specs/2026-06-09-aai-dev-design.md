# `aai dev` — launch a scaffolded template's dev server

**Date:** 2026-06-09
**Status:** Approved (design)

## Problem

After `aai init` scaffolds a template, re-running the app means remembering and
hand-typing `uvicorn api.index:app --reload --port 3000`. Every template README
and `AGENTS.md` documents that raw command, and `aai init`'s sign-off hints point
at it too. We want a first-class `aai dev` that "just works" from inside a
scaffolded project, and we want the docs/hints to advertise `aai dev` instead.

## Goals

- `aai dev`, run from a scaffolded template directory, installs dependencies if
  needed and launches the FastAPI dev server with live reload, opening the
  browser.
- Replace the hand-typed `uvicorn …` command everywhere it's advertised
  (template `README.md` + `AGENTS.md`, `aai init` hints) with `aai dev`.

## Non-goals

- Detecting *which* template is running. All three templates share the identical
  serve contract (`api/index:app`), so `dev` is template-agnostic.
- Walking up parent directories to find a project root. Detection is
  current-directory-only (see Decisions).
- Production serving / deploy. `dev` is a local development server only.

## Decisions

1. **Detection: current directory only.** `aai dev` runs iff
   `Path.cwd() / "api" / "index.py"` exists. No parent-directory walk, no
   per-template marker file. The user must be at the project root (the directory
   `aai init` created / `--here` targeted).
2. **Auto-install then launch.** If dependencies aren't set up, `dev` runs the
   existing `runner.run_setup` (uv `venv` + `uv pip install -r requirements.txt`,
   or stdlib `venv` + `pip` when uv is absent) before launching. A `--no-install`
   flag skips this for users who manage their own environment.
3. **Live reload on by default.** `dev` always passes `--reload` to uvicorn so
   edits hot-reload. This matches what every template README already documents.
   `aai init`'s launch path keeps reload **off** (unchanged behavior).

## Architecture

### `aai_cli/init/runner.py` (one additive change)

Thread a `reload` flag through the serve path:

- `serve_command(target, *, port, use_uv, reload=False)` — append `--reload` to
  the uvicorn argv when `reload` is true. Default `False` preserves `init`'s
  current behavior with no change to its call site.
- `launch_and_open(target, *, port, use_uv, open_browser, reload=False)` —
  forward `reload` to `serve_command`.

Everything else in `runner.py` (`run_setup`, `find_free_port`, `wait_for_port`,
`has_uv`) is reused as-is.

### `aai_cli/commands/dev.py` (new)

Single flattened Typer command, mirroring `init.py`'s
`app = typer.Typer()` + `app.add_typer(..., no name)` pattern.

Flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | `3000` | Local server port (first free port at/above this). |
| `--no-open` | `False` | Launch but don't open the browser. |
| `--no-install` | `False` | Skip the auto-install step; launch directly. |
| `--json` | `False` | Machine-readable output. |

Command body (run through `context.run_command`):

1. **Locate the app.** `app_file = Path.cwd() / "api" / "index.py"`. If it doesn't
   exist → `CLIError(error_type="usage_error", exit_code=1)` whose message tells
   the user to `cd` into a directory created by `aai init`, or run `aai init`
   first.
2. **Resolve runner.** `use_uv = runner.has_uv()`.
3. **Install (unless `--no-install`).** `runner.run_setup(cwd, use_uv=use_uv)`.
   On non-zero return, emit a failed `install` step and exit non-zero (mirrors
   `init`'s `_install_step` failure shape). Emit an `installed`/`skipped` step
   otherwise.
4. **Launch.** `port = runner.find_free_port(port)`; print the
   `Starting http://localhost:PORT  (Ctrl-C to stop)` banner (non-JSON only,
   reusing `init`'s `_launch` styling); call
   `runner.launch_and_open(cwd, port=port, use_uv=use_uv, open_browser=not no_open, reload=True)`.
   Propagate a non-zero server exit code via `typer.Exit`.

Because the install/launch styling already exists in `init.py`, the shared bits
(`_launch` banner, `_install_step` failure row) are small enough to duplicate
cleanly in `dev.py` rather than extracting a shared helper prematurely; if a
third caller appears, extract then.

### `aai_cli/main.py` (registration)

- Add `dev` to the `from aai_cli.commands import (...)` block.
- `app.add_typer(dev.app)` alongside the other sub-apps.
- Insert `"dev"` into `_COMMAND_ORDER` immediately after `"init"`.
- File `dev` under the **Build an App** panel via
  `rich_help_panel=help_panels.BUILD` on the command.

## Documentation / hint changes

Replace the advertised raw command with `aai dev`:

- `aai_cli/init/templates/audio-transcription/README.md`
- `aai_cli/init/templates/live-captions/README.md`
- `aai_cli/init/templates/voice-agent/README.md`
- `aai_cli/init/templates/audio-transcription/AGENTS.md`
- `aai_cli/init/templates/live-captions/AGENTS.md`
- `aai_cli/init/templates/voice-agent/AGENTS.md`

In each "Run locally" block, the
`uvicorn api.index:app --reload --port 3000` / `# open http://localhost:3000`
lines become a single `aai dev` line (keeping a note that it serves on
`http://localhost:3000` and reads `ASSEMBLYAI_API_KEY` from `.env`).

`aai_cli/commands/init.py` two hints:

- The "no API key" launch-skipped detail:
  `… run \`aai login\`, then: cd {target} && aai dev`.
- The scaffold-only sign-off hint:
  `Run \`cd {target} && aai dev\`.`

## Errors

- Not in a template dir → `usage_error`, exit 1, actionable message.
- Install failure → failed step row + exit 1 (no server start).
- Clean Ctrl-C → exit 0 (handled by `launch_and_open`'s `KeyboardInterrupt`).
- Non-zero server exit → propagated as the command's exit code.

## Testing

New `tests/test_dev.py`, mocking at the `runner` boundary per the repo's
pytest-mock convention (no real subprocess / network):

- Launch happy path: `api/index.py` present, install succeeds, `launch_and_open`
  called with `reload=True` and the resolved free port; browser opened unless
  `--no-open`.
- `--no-open` → `open_browser=False`.
- `--no-install` → `run_setup` not called; still launches.
- Missing `api/index.py` → `usage_error`, exit 1, `run_setup`/`launch_and_open`
  never called.
- Install failure → exit 1, `launch_and_open` never called.
- `--json` path emits structured steps.

Gate consequences to handle in the same change:

- Regenerate the `aai --help` syrupy snapshot (`--snapshot-update`) — a new
  command and panel entry appears.
- Update any template-contract test (`tests/test_init_template_*.py`) that pins
  the old `uvicorn …` string in README/AGENTS content.
- Maintain ≥90% branch coverage and 100% patch coverage (`diff-cover`).

## Out of scope / future

- Parent-directory project-root discovery.
- A `--reload/--no-reload` toggle (reload is unconditional for now).
- Reusing `dev`'s launch from inside `aai init` (init keeps its own non-reload
  launch).
