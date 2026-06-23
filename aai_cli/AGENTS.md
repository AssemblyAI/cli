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
  `config_builder`, `keyring_store`, `environments`, `env`, `errors`, `llm`,
  `telemetry`, `debuglog`, `remotefs`, `sync_stt`, `signals`, `ws`, `youtube`,
  `wer`, `argscan`, `jsonshape`, `timeparse`, `microphone`, `procs`, `stdio`,
  `choices`, `ssrf` (the outbound-fetch SSRF guard: resolves a URL's host and
  refuses private/loopback/link-local IPs, re-checked on every redirect hop —
  shared by `webpage` and `app.transcribe.feed`). Contract 4 also forbids `rich`
  here, so "no Rich below the UI layer" is structural.

Three things sit *beside* the stack, intentionally unlisted in the layers
contract:

- **CLI framework glue at the package root** — `main`, `command_registry`,
  `help_panels`, `options`. They assemble/define the command layer (and
  `command_registry` imports the command modules to discover them), so they live
  *above* `commands` and stay at the root.
- **Feature slices** — `agent/`, `tts/`, `streaming/`, `code_gen/`,
  `init/`, `auth/`, `onboard/`. These are cohesive vertical slices that internally mix
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
- **`core/config.py`** — profiles persisted in `config.toml` (via `platformdirs`); the **API key lives only in the OS keyring**, never in a dotfile. The keyring access itself is factored into **`core/keyring_store.py`** (the single importer of `keyring`, holding `KEYRING_SERVICE = "assemblyai-cli"` + `set_secret`/`get_secret`/`restore_secret`/`delete_secret`/`usable`), so the "secrets never touch the dotfile" split is structural; `config` reads/writes secrets through it and only `config.keyring_usable` re-surfaces the probe on the auth facade. Key resolution order: `--api-key` flag (validation paths only) → `ASSEMBLYAI_API_KEY` env → keyring. **Run commands deliberately expose no `--api-key` flag** so keys can't leak into `ps`/shell history. The `config.toml` document schema (`Profile`/`Config`/`StoredSession`) and its parse/cache/atomic-write machinery live one layer down in **`core/config_store.py`** (the same factoring as `keyring_store`): `config` is the auth/profile facade and reads as plain accessors over `config_store.load`/`dump`/`update`, so the store rules stay structural. Every `config.toml` write is a read-modify-write (`load` → mutate → `dump`) via the `config_store.update` context manager: `dump` is a temp-file + atomic `os.replace`, so a reader never sees a torn file. Writers and readers are otherwise unsynchronized — last write wins (there is **no** cross-process lock; an earlier `filelock`-based serialization was removed because it was a recurring Windows CI flake and the lost-update race it closed isn't worth the cost for a single-user CLI). On Windows the atomic replace has no replace-over-open guarantee, so both the lock-free read and the `os.replace` ride out the transient `PermissionError` through `config_store._retry_on_sharing_violation` (a no-op on POSIX). Tests isolate the config dir by patching `config_store.config_dir` (the autouse `tmp_config` fixture).
- **`core/environments.py`** — a frozen `Environment` (api_base, streaming_host, llm_gateway_base, ams_base, stytch_*). `DEFAULT_ENV` is **`production`**; use `--sandbox` (or `--env sandbox000` / `AAI_ENV`) to target the sandbox. The active environment is a process-global set once at startup; precedence: `--env` → `AAI_ENV` → profile's stored env → default. A credential is only valid against the environment that minted it.
- **`core/client.py`** — thin wrappers over the `assemblyai` SDK (`transcribe`, `list_transcripts`, `stream_audio`, etc.). It normalizes SDK exceptions: auth failures become a single clean `auth_failure()` `CLIError`; everything else becomes `APIError`. New SDK calls should follow this try/except shape.
- **`core/errors.py`** — the `CLIError` hierarchy (each with `error_type` + `exit_code`). `ui/output.py` emits errors to **stderr**; stdout stays clean for pipelines. `--json` switches to machine-readable output; it is never auto-enabled — `output.resolve_json()` deliberately keeps human text the default even when piped or agent-run.
- **Raw `subprocess` and `os.environ`/`os.getenv` are fenced by ruff `banned-api` (TID251).** Environment access has a single chokepoint: **`core/env.py`** is the only module allowlisted for raw `os.environ` — every other module reads/writes the environment through `env.get`/`env.child_env`/`env.force_color`/… (callers still own their variable *names*, e.g. `config.ENV_API_KEY`). Process spawning is the sibling boundary, but unlike env reads it's genuinely diverse (sync-capture, long-lived `Popen` with pipes, detached children), so each module that shells out to its specific tool stays individually allowlisted rather than funnelling through one module. A new module reaching past either boundary trips the gate, so adding one is a deliberate, reviewable edit (the Deno toolchain's per-crate `clippy.toml` model). Tests and `scripts/` are exempt.
- **`core/debuglog.py`** — the root `-v/--verbose` flag (count: `-v` request-level at INFO, `-vv` wire-level at DEBUG). The CLI normally configures no logging, and the realtime paths *silence* library loggers (`ws.py`, `streaming/diagnostics.py`); verbose mode installs one redacting stderr handler and those silencers stand down. Secrets are registered at their resolution choke points (`config.resolve_api_key`, `AppState.resolve_session`) and masked in every rendered record — websockets logs the raw Authorization header at DEBUG, so masking lives in the formatter, not at call sites. Stdlib-only on purpose: `config` (a Rich-free layer) imports it.

### Feature subsystems

- **`streaming/`** + `client.stream_audio` — v3 realtime API. Event callbacks run on the SDK reader thread and guard against `BrokenPipeError` (`stdio.silence_stdout()`) so a closed pipe never dumps a thread traceback.
- **`core/sync_stt.py`** + **`core/signals.py`** + `commands/dictate/` — `assembly dictate`: headless dictation over the **Sync STT API** (`Environment.sync_base`, one POST `/transcribe` per utterance with the required `X-AAI-Model: u3-sync-pro` header; 80 ms–120 s of PCM/WAV). It needs no terminal: recording starts immediately and `dictate_exec._record` polls `signals.stop_on_terminate` between ~100 ms mic chunks for a SIGTERM, which finishes the utterance (clean exit 0) — so a hotkey tool like Hammerspoon can launch it as a background task and `kill -TERM`/`task:terminate()` to transcribe. SIGINT (Ctrl-C) still cancels (exit 130). Both boundaries (the stop latch, mic, HTTP) are injectable, so the suite never needs a real signal or microphone (`tests/test_dictate_exec.py` scripts the SIGTERM latch). Contrast `signals.terminate_as_interrupt` (used by `stream`/`agent`/`speak`), which routes SIGTERM into the *cancel* path instead.
- **`agent/`** — full-duplex voice agent (mic in, TTS out via `voices.py`).
- **`agent_cascade/`** + `commands/agent_cascade/` — `assembly agent-cascade`: the same live terminal conversation as `assembly agent`, but **client-orchestrated** — `engine.run_cascade` wires Streaming STT → the LLM Gateway → streaming TTS itself instead of talking to the Voice Agent endpoint, mirroring what the `agent-cascade` `assembly init` template does server-side. **Sandbox-only** (streaming TTS has no prod host; guarded via `tts.session.require_available`). Reuses the agent slice's `DuplexAudio`/`AgentRenderer` and `core.client.stream_audio`/`core.llm.complete`/`tts.session.synthesize`; the three network legs are injected through `engine.CascadeDeps` (the `tts/session.py` seam) so the cascade — greeting, clause-level streaming TTS, barge-in — is unit-tested against fakes with no sockets/mic/speaker. The LLM leg is a deepagents graph (`brain.py`) streamed token-by-token via `brain.build_streamer` (`graph.stream(stream_mode="messages")`): **context-window management is the brain's job, not the engine's** — `create_deep_agent` wires deepagents' own `SummarizationMiddleware` into the stack (summarize the oldest turns, offload the evicted history to a file), so the engine feeds the *full* untrimmed running history each turn and lets the graph compact it; the old client-side `text.trim_history`/`config.max_history` sliding window is gone from this path (`max_history` now only drives the hand-rolled `--show-code`/`assembly init` cascade, which doesn't use deepagents). The engine buffers `SpeechDelta`s, flushes complete clauses with `text.pop_clauses` (soft-separator clauses gated by `engine._MIN_CLAUSE_CHARS`), and synthesizes each clause with **streaming TTS** (`tts.session.synthesize(on_audio=…)`) so audio starts on the first frame instead of after the whole reply. The reply runs on a throwaway producer thread feeding a `queue.Queue` the worker drains under a monotonic deadline (the wall-clock backstop that replaced `_complete_within`), and an abandoned-on-timeout graph leg's langchain `ThreadPoolExecutor` worker is detached (`_detach_executor_threads_since`) so it can't wedge interpreter exit. A `ToolNotice` surfaces the "Searching the web…" affordance and drops any unspoken preamble. Under `-v` (`debuglog.active()`) `brain._stream_graph` logs each accumulated assistant line, tool call, and tool result as it streams. **Front-end:** an interactive mic session in human mode runs a **voice-only Textual TUI** (`agent_cascade/tui.py`, `LiveAgentApp`) by default — there's no text input (you can't type to it), just a transcript + an animated voice bar tracking listening/thinking/speaking. It uses its own `banner` wordmark, `messages` widgets, and `tui_status.voicebar_markup`/`VOICE_FRAMES` — all modules that now live in `agent_cascade/`; the blocking `run_cascade` runs on a worker thread and reaches the UI through a `_TuiRenderer` (the `engine.Renderer` protocol) that hops each call onto the UI thread, and a quit calls `DuplexAudio.close` to end the mic iterator and unblock that worker. `_exec._should_use_tui` gates it: file/sample input, `--json`/`-o text`, and a non-TTY all fall back to the plain `AgentRenderer` line output. **`--files`** (on by default; `--no-files` opts out) swaps the brain's in-memory backend for a real-cwd, sandbox-capable `SandboxedShellBackend` (`aai_cli/agent_cascade/sandbox.py`): file ops behave as before (traversal-blocked `virtual_mode`), and because it implements `SandboxBackendProtocol` deepagents binds a *functional* `execute` that runs commands OS-sandboxed in the real cwd — `sandbox-exec` (SBPL) on macOS, `bwrap` on Linux, refused (never an unconfined fallback) on any other platform or with the sandbox binary missing; the OS sandbox blocks the network, confines writes to cwd (+ the temp dir), and read-denies credential stores (`~/.ssh`/`~/.aws`/…, `.env*`, `.claude/`). The policy renderers are pure and the subprocess/capability boundaries injected, so the suite asserts *what we'd run* with no real sandbox. `write_file`/`edit_file`/`execute` are gated via `interrupt_on` + an `InMemorySaver`; `brain._stream_gated` detects the post-stream interrupt (`graph.get_state(config).interrupts`), asks an injected `Approver`, and resumes with `Command(resume=…)`, bracketing the human wait in `ApprovalPause` events so `engine._consume` suspends its reply deadline (`risk.py` surfaces a shell-risk warning on the prompt). The voice TUI supplies the approver via `agent_cascade.modals.ApprovalScreen` (`y`/`a`/`n`), which can *also* be resolved hands-free by voice: while a write awaits approval, `_consume` arms `_awaiting_approval` and `engine.on_turn` routes the next final transcript to `app.submit_voice_approval` → `ApprovalScreen.try_voice`, which applies `spoken_approval.spoken_decision` (an unambiguous affirmative approves, anything else rejects — fail-safe; destructive `risk.py`-flagged commands ignore the spoken answer and require a keypress). Headless runs auto-deny (`_exec._deny_writes`). `--files` also turns on durable per-project memory via deepagents' `MemoryMiddleware` (`memory=["./.deepagents/AGENTS.md"]`), distinct from the in-session `InMemorySaver`. The gateway-bound, sandbox-backed general-purpose subagent (deepagents' `task` tool) for delegating a focused subtask is **auto-added by deepagents** — we don't declare it. We only override its prose for a voice turn (a spoken-length summary, not the SDK's "complete answer" default) via a harness profile keyed by the gateway model's provider (`subagents.register_gp_subagent_profile`, called from `build_graph` so the deepagents import stays lazy — and kept off `brain.py`, which sits at the 500-line gate). It inherits the gateway-bound model, the sandboxed toolset, *and* the top-level `interrupt_on` (deepagents' `graph.py` merges the top-level config into the auto-added subagent), so a delegated `write_file`/`edit_file`/`execute` surfaces at the *parent* `get_state().interrupts` with no per-subagent restatement (so `_pending_writes` gates it too — verified by a HITL spike, locked in `tests/test_agent_cascade_subagents.py`). Reads (incl. `grep`) stay ungated.
- **`tts/`** + `commands/speak.py` — `assembly speak` synthesizes text to speech over the sandbox streaming-TTS WebSocket (`streaming-tts.sandbox000.…`). **Sandbox-only:** `session.is_available()` is false in production (empty `Environment.streaming_tts_host`), so the command exits 2 with a `--sandbox` hint. `session.synthesize` drives a Begin→Generate→Flush→Audio→Terminate protocol with an injectable `connect` for hermetic tests (mirrors `agent/session.py`); `audio.py` plays the PCM (default) or writes a WAV (`--out`). The single-voice default-playback path **streams**: `synthesize`'s `on_audio(chunk, sample_rate)` callback is wired to `audio.PcmPlayer.feed`, so speech starts on the first Audio frame (it opens the device lazily, since the rate is only known at Begin) instead of after the whole text — the win for a long `--url` page. `--out` (needs the full buffer) and the multi-voice dialogue path (`synthesize_dialogue` → `_output_audio` → buffered `play_pcm`) stay buffered; `synthesize` still returns the complete PCM for the summary regardless.
- **`code_gen/`** — backs `--show-code` on `transcribe`/`stream`/`agent`: builds a ready-to-run Python SDK script from exactly the flags passed (no API key needed; generated code reads `ASSEMBLYAI_API_KEY`).
- **`auth/`** — browser-assisted `assembly login` via AMS + **Stytch B2B OAuth discovery** (`discovery.py`, `flow.py`, `loopback.py`, `ams.py`). Not Stytch Connected Apps.
- **`init/`** — scaffolds a self-contained FastAPI + HTML starter (`audio-transcription`/`live-captions`/`voice-agent` templates), optionally installs deps and opens the browser; writes the key to a git-ignored `.env`.
- **`core/telemetry.py`** — anonymous, opt-out usage telemetry (Supabase-CLI model): `context.run_command` wraps each command body in `telemetry.track(ctx.command_path)`, which dispatches one allow-listed event (command path, outcome/exit code, duration, version/OS, and on failure the error message capped at 500 chars — never args or account data) to the Datadog logs intake via a **detached flusher subprocess** (the hidden `assembly telemetry flush`), so commands never wait on telemetry. `SHIPPED_CLIENT_TOKEN` is a committed write-only Datadog *client* token (`pub…`, embeddable by design — never an API key; `AAI_TELEMETRY_CLIENT_TOKEN` overrides). The test suite blanks it via an autouse conftest fixture so no test ever spawns a real flusher. Opt-out: `AAI_TELEMETRY_DISABLED=1` / `DO_NOT_TRACK=1` / `assembly telemetry disable` (persisted as `telemetry_enabled` in config.toml, alongside the random `device_id`). Send-side failures are swallowed (`OSError`/`CLIError`) — telemetry must never break a command.
- **`commands/setup.py`** + **`app/setup_exec.py`** — `assembly setup install/status/remove` wires a coding agent up to AssemblyAI by installing three artifacts: the `assemblyai-docs` docs MCP (via `claude mcp add`), the AssemblyAI skill (via `npx skills add`), and the bundled `aai-cli` skill (copied out of the wheel, no network). Missing `claude`/`npx` is reported and skipped, not an error. The step implementations live in `aai_cli/app/setup_exec.py` and the presence probes (docs MCP registered, skills on disk) in `aai_cli/app/coding_agent.py`, so `assembly doctor` (via `app/doctor_checks.py`) and the onboarding wizard share them without command modules importing each other.
