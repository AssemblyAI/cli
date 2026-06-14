# aai_cli/ — architecture guide

Scoped guidance for the package source. Repo-wide invariants (gate, commit
hooks, conventions) live in the root `AGENTS.md`; test-suite guidance lives in
`tests/AGENTS.md`.

## Architecture

A Typer CLI. `aai_cli/main.py` builds the `app` and registers every command
module discovered by `aai_cli/command_registry.py`. Typer/Click/Rich overrides
(help palette, column clipping, pipe-safe consoles, Click error formatting)
live in `aai_cli/ui/typer_patches.py` — one file to fix when a dependency
upgrade breaks a patch; each patch documents the upstream behavior it overrides.
`run()` is the entry point and swallows `BrokenPipeError` (closed downstream
pipe → exit 0).

### Package layout (layered)

The package is organized as a layered stack, enforced by `.importlinter`
contract 1 (`type = layers`, `commands > app > ui > core`). Each layer is a
single package, so imports *within* a layer are free and only the *direction*
between layers is enforced — higher may import lower, never the reverse:

- **`commands/`** — the Typer sub-apps (top of the stack; see the convention
  below).
- **`app/`** — orchestration / shared run-logic that wires features together and
  is reused beyond one command: `context`, the `transcribe/` subpackage
  (`run`/`render`/`batch`/`sources`/`validate`), `init_exec`, `setup_exec`,
  `doctor_checks`, `coding_agent`, `mediafile` (it renders via the UI layer, so
  it sits here, not in `core`).
- **`ui/`** — Rich rendering: `output`, `render`, `theme`, `steps`, `follow`,
  `help_text`, `typer_patches`, `update_check`.
- **`core/`** — the Rich-free library layer: `client`, `config`,
  `config_builder`, `environments`, `env`, `errors`, `llm`, `telemetry`,
  `debuglog`, `remotefs`, `sync_stt`, `hotkey`, `ws`, `youtube`, `wer`,
  `argscan`, `jsonshape`, `timeparse`, `microphone`, `procs`, `stdio`,
  `choices`. Contract 4 also forbids `rich` here, so "no Rich below the UI
  layer" is structural.

Three things sit *beside* the stack, intentionally unlisted in the layers
contract:

- **CLI framework glue at the package root** — `main`, `command_registry`,
  `help_panels`, `options`. They assemble/define the command layer (and
  `command_registry` imports the command modules to discover them), so they live
  *above* `commands` and stay at the root.
- **Feature slices** — `agent/`, `tts/`, `streaming/`, `code_gen/`, `init/`,
  `auth/`, `onboard/`. These are cohesive vertical slices that internally mix
  protocol + rendering, so they aren't a single horizontal layer; contract 2
  forbids them from importing `commands`.

A new top-level module must land in one of these buckets;
`tests/test_importlinter_coverage.py` fails loudly if one escapes the partition.
The intra-layer split is invisible to importers in the *same* layer, but always
import across layers by the full path (`from aai_cli.core import config`,
`from aai_cli.ui import output`, `from aai_cli.app.context import AppState`).

### Command layer & the registration convention

Each entry under `aai_cli/commands/` is a Typer sub-app (`transcribe`, `stream`,
`dictate`, `agent`, `speak`, `llm`, `clip`, `dub`, `caption`, `eval`,
`transcripts`, `login` (login/logout/whoami), `doctor`, `init`, `dev`, `share`,
`deploy`, `setup`, `onboard`, `account` (balance/usage/limits), `keys`,
`sessions`, `audit`, `telemetry` (status/enable/disable), `webhooks` (listen)).

**A command is either a single module *or* a package** — `command_registry`
discovers both (it iterates `pkgutil.iter_modules`, which enumerates packages
too). A simple command stays a flat `commands/<cmd>.py`. A command with private
run-logic becomes a package `commands/<cmd>/`: `__init__.py` holds the Typer
`app` + `SPEC` (and is what gets imported as `aai_cli.commands.<cmd>`), and its
support modules sit beside it **underscore-prefixed** — `_exec.py` for the
`run_<cmd>` body, plus any private helpers (`clip/_select.py`,
`evaluate/_data.py`, `evaluate/_hf_api.py`). The underscore both marks them
private and avoids colliding with the package's own command functions (the
`webhooks` package binds a `listen` command, so its module is `_listen.py`, not
`listen.py`). This is the Prefect/spaCy convention: flat file by default,
promote to a folder only when the command has earned multiple modules. Run-logic
that's **shared beyond one command lives in the `app/` layer**, not inside a
command package — the `app/transcribe/` subpackage (`run`/`render`/`batch`/
`sources`/`validate` — promoted from flat `transcribe_*` modules once the family
outgrew one file) and `app/init_exec` are reused by the onboarding wizard
(`onboard/sections.py`), so they live in `app/` alongside
`doctor_checks`/`setup_exec` rather than under `commands/transcribe/` or
`commands/init/`.

**Adding a command is purely additive — no shared file edits.** Every command
module declares a module-level
`SPEC = command_registry.CommandModuleSpec(panel=…, order=…, commands=…)`:

- `panel`: one of `help_panels.PANEL_ORDER` — which `assembly --help` panel its
  commands render under. This declaration also derives the help-snapshot
  partition (`HELP_GROUPS` in `tests/_snapshot_surface.py`), so a new command
  is automatically required to have a `--help` golden in the right group.
- `order`: a sparse rank within the panel (10, 20, 30, …) so a new command
  slots between neighbors without renumbering them. Mark the line
  `# pragma: no mutate` — a ±1 shift is order-equivalent, so no test can kill
  that mutant.
- `commands`: the top-level command names the module contributes, in display
  order (multi-command merged modules like `login` list all three).
- `group_name`: set for named sub-groups (`assembly keys list` style); the
  registry then passes it to `add_typer(name=…, rich_help_panel=…)`. Merged
  (nameless) modules instead set `rich_help_panel` on each `@app.command()`.

`command_registry.discover()` imports every module under `aai_cli/commands/`,
validates the convention (a module missing `SPEC` or `app` fails loudly at
import), and orders them; `main.py` registers the result. The help ordering,
the root `--help` golden, and the snapshot partition are all derived from the
same `SPEC`s.

Command bodies run through `context.run_command(ctx, fn, json=...)`, which maps
any `CLIError` to clean stderr output + the error's exit code. Commands never
print tracebacks for expected failures.

**Command modules are import-linter-independent** (`.importlinter` contract 3,
wildcarded over `aai_cli.commands.*` so new modules are covered automatically).
Logic shared between commands lives in the `app/` layer: `app/doctor_checks.py`
(diagnostics shared by `doctor` and onboarding) and `app/setup_exec.py`
(installer steps shared by `setup` and onboarding) are the precedent — never
import one command module from another.

**Options/run split for flag-heavy commands** (gh-CLI style): the Typer
function only parses argv into a frozen `<Cmd>Options` dataclass and hands it
to a module-level `run_<cmd>(opts, state, *, json_mode)` via
`context.run_with_options(ctx, run_<cmd>, opts, json=...)` — the typed adapter
that wraps the `run_<cmd>` body in the `(state, json_mode)` callable
`run_command` expects, so no command repeats the `lambda state, json_mode: …`
boilerplate. The run commands follow it —
`commands/stream/_exec.py` (the reference implementation), `app/transcribe/run.py`
(in the `app/` layer — shared with onboarding), `commands/agent/_exec.py`,
`commands/speak/_exec.py`, `commands/llm/_exec.py`, `commands/clip/_exec.py`,
`commands/dictate/_exec.py`. Because the run path is a plain function of data, tests
construct options directly (`dataclasses.replace` off a defaults instance, see
`tests/test_stream_exec.py` and `tests/test_command_options_seam.py`) instead
of round-tripping argv through `CliRunner` — which is also the cheap way to
kill mutation-gate mutants on orchestration lines. Follow this for new or
heavily-reworked commands with long bodies; small commands keep the inline
`body()` closure — the dataclass is pure ceremony there.

### Cross-cutting state (resolution order matters)

- **`app/context.py`** — `AppState` (profile, env) is attached to the Typer context in the root `@app.callback()`. `run_command` is the standard command wrapper.
- **`core/config.py`** — profiles persisted in `config.toml` (via `platformdirs`); the **API key lives only in the OS keyring** (`KEYRING_SERVICE = "assemblyai-cli"`), never in a dotfile. Key resolution order: `--api-key` flag (validation paths only) → `ASSEMBLYAI_API_KEY` env → keyring. **Run commands deliberately expose no `--api-key` flag** so keys can't leak into `ps`/shell history.
- **`core/environments.py`** — a frozen `Environment` (api_base, streaming_host, llm_gateway_base, ams_base, stytch_*). `DEFAULT_ENV` is **`production`**; use `--sandbox` (or `--env sandbox000` / `AAI_ENV`) to target the sandbox. The active environment is a process-global set once at startup; precedence: `--env` → `AAI_ENV` → profile's stored env → default. A credential is only valid against the environment that minted it.
- **`core/client.py`** — thin wrappers over the `assemblyai` SDK (`transcribe`, `list_transcripts`, `stream_audio`, etc.). It normalizes SDK exceptions: auth failures become a single clean `auth_failure()` `CLIError`; everything else becomes `APIError`. New SDK calls should follow this try/except shape.
- **`core/errors.py`** — the `CLIError` hierarchy (each with `error_type` + `exit_code`). `ui/output.py` emits errors to **stderr**; stdout stays clean for pipelines. `--json` switches to machine-readable output; it is never auto-enabled — `output.resolve_json()` deliberately keeps human text the default even when piped or agent-run.
- **Raw `subprocess` and `os.environ`/`os.getenv` are fenced by ruff `banned-api` (TID251).** Environment access has a single chokepoint: **`core/env.py`** is the only module allowlisted for raw `os.environ` — every other module reads/writes the environment through `env.get`/`env.child_env`/`env.force_color`/… (callers still own their variable *names*, e.g. `config.ENV_API_KEY`). Process spawning is the sibling boundary, but unlike env reads it's genuinely diverse (sync-capture, long-lived `Popen` with pipes, detached children), so each module that shells out to its specific tool stays individually allowlisted rather than funnelling through one module. A new module reaching past either boundary trips the gate, so adding one is a deliberate, reviewable edit (the Deno toolchain's per-crate `clippy.toml` model). Tests and `scripts/` are exempt.
- **`core/debuglog.py`** — the root `-v/--verbose` flag (count: `-v` request-level at INFO, `-vv` wire-level at DEBUG). The CLI normally configures no logging, and the realtime paths *silence* library loggers (`ws.py`, `streaming/diagnostics.py`); verbose mode installs one redacting stderr handler and those silencers stand down. Secrets are registered at their resolution choke points (`config.resolve_api_key`, `AppState.resolve_session`) and masked in every rendered record — websockets logs the raw Authorization header at DEBUG, so masking lives in the formatter, not at call sites. Stdlib-only on purpose: `config` (a Rich-free layer) imports it.

### Feature subsystems

- **`streaming/`** + `client.stream_audio` — v3 realtime API. Event callbacks run on the SDK reader thread and guard against `BrokenPipeError` (`stdio.silence_stdout()`) so a closed pipe never dumps a thread traceback.
- **`core/sync_stt.py`** + **`core/hotkey.py`** + `commands/dictate/` — `assembly dictate`: push-to-talk dictation over the **Sync STT API** (`Environment.sync_base`, one POST `/transcribe` per utterance with the required `X-AAI-Model: u3-sync-pro` header; 80 ms–120 s of PCM/WAV). `hotkey.TerminalKeys` scopes stdin into cbreak (Ctrl-C still signals) and reads single keypresses; `dictate_exec._record` polls it with a zero timeout between ~100 ms mic chunks. All three boundaries (keys, mic, HTTP) are injectable, so the suite never needs a real terminal — `tests/test_hotkey.py` drives a pty pair for the termios behavior.
- **`agent/`** — full-duplex voice agent (mic in, TTS out via `voices.py`).
- **`tts/`** + `commands/speak.py` — `assembly speak` synthesizes text to speech over the sandbox streaming-TTS WebSocket (`streaming-tts.sandbox000.…`). **Sandbox-only:** `session.is_available()` is false in production (empty `Environment.streaming_tts_host`), so the command exits 2 with a `--sandbox` hint. `session.synthesize` drives a Begin→Generate→Flush→Audio→Terminate protocol with an injectable `connect` for hermetic tests (mirrors `agent/session.py`); `audio.py` plays the PCM (default) or writes a WAV (`--out`).
- **`code_gen/`** — backs `--show-code` on `transcribe`/`stream`/`agent`: builds a ready-to-run Python SDK script from exactly the flags passed (no API key needed; generated code reads `ASSEMBLYAI_API_KEY`).
- **`auth/`** — browser-assisted `assembly login` via AMS + **Stytch B2B OAuth discovery** (`discovery.py`, `flow.py`, `loopback.py`, `ams.py`). Not Stytch Connected Apps.
- **`init/`** — scaffolds a self-contained FastAPI + HTML starter (`audio-transcription`/`live-captions`/`voice-agent` templates), optionally installs deps and opens the browser; writes the key to a git-ignored `.env`.
- **`core/telemetry.py`** — anonymous, opt-out usage telemetry (Supabase-CLI model): `context.run_command` wraps each command body in `telemetry.track(ctx.command_path)`, which dispatches one allow-listed event (command path, outcome/exit code, duration, version/OS, and on failure the error message capped at 500 chars — never args or account data) to the Datadog logs intake via a **detached flusher subprocess** (the hidden `assembly telemetry flush`), so commands never wait on telemetry. `SHIPPED_CLIENT_TOKEN` is a committed write-only Datadog *client* token (`pub…`, embeddable by design — never an API key; `AAI_TELEMETRY_CLIENT_TOKEN` overrides). The test suite blanks it via an autouse conftest fixture so no test ever spawns a real flusher. Opt-out: `AAI_TELEMETRY_DISABLED=1` / `DO_NOT_TRACK=1` / `assembly telemetry disable` (persisted as `telemetry_enabled` in config.toml, alongside the random `device_id`). Send-side failures are swallowed (`OSError`/`CLIError`) — telemetry must never break a command.
- **`commands/setup.py`** + **`app/setup_exec.py`** — `assembly setup install/status/remove` wires a coding agent up to AssemblyAI by installing three artifacts: the `assemblyai-docs` docs MCP (via `claude mcp add`), the AssemblyAI skill (via `npx skills add`), and the bundled `aai-cli` skill (copied out of the wheel, no network). Missing `claude`/`npx` is reported and skipped, not an error. The step implementations live in `aai_cli/app/setup_exec.py` and the presence probes (docs MCP registered, skills on disk) in `aai_cli/app/coding_agent.py`, so `assembly doctor` (via `app/doctor_checks.py`) and the onboarding wizard share them without command modules importing each other.
