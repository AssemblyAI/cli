# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, and
others) when working with code in this repository. `CLAUDE.md` is a symlink to
this file, so Claude Code reads the same instructions.

## Development commands

This project uses [uv](https://docs.astral.sh/uv/). **Run every Python tool through `uv run`** so it uses the locked environment (`pyproject.toml` + `uv.lock`), not whatever is on `PATH`:

```sh
uv sync                      # create/refresh the venv (the dev group installs by default)
uv run assembly --help            # run the CLI from the locked environment
./scripts/check.sh           # the full gate CI runs (scripts/check.sh is the source of truth)
```

Dev tooling is a PEP 735 `[dependency-groups]` group with `default-groups = ["dev"]`, not a `[project]` extra — `uv sync --extra dev` errors.

`scripts/check.sh` is the authoritative gate; keep this list in sync with it. It runs, in order: `uv lock --check` → `ruff check` → `ruff format --check` → `mypy` → `pyright` (src strict) → `pyright` (tests) → `vulture` (dead code) → `deptry` (dependency hygiene) → `lint-imports` (import-linter architecture contracts) → max-file-length (500 lines) → `xenon` (cyclomatic complexity, max grade B / project avg A) → `swiftlint` + swift compile (macOS only, skipped elsewhere) → `markdownlint` → `prettier` (init template JS/CSS) → `shellcheck` → `actionlint` + `zizmor` (workflow lint/audit) → `gitleaks` (secret scan) → generated `--show-code` compile gate → init template contract gate → `pytest` (90% branch coverage) → `diff-cover` (100% patch coverage vs `origin/main`) → **mutation gate** (diff-scoped: mutates each changed line and reruns the tests that cover it — a surviving mutant fails the gate, so changed lines need assertions that would *fail* if the line broke, not just coverage; suppress a genuinely unassertable line with `# pragma: no mutate`) → a "no new escape hatches" diff gate (`# type: ignore` / `# noqa` / `pragma: no cover` / net-new `Any` / `cast(`) → `uv build` + `twine check --strict`. The `vulture`/`deptry`/`lint-imports`/`xenon`, patch-coverage, and mutation stages catch the failures that `ruff`+`mypy` alone won't — don't claim the gate is green until the script prints `All checks passed.`

**Commits are gated.** On success `check.sh` records a working-tree signature (`scripts/gate_marker.py record` → `.git/aai-gate-pass`), and a PreToolUse hook (`.claude/hooks/require-gate-before-commit.sh`) blocks `git commit` unless that signature still matches — so run the full gate to completion *before* committing (a single-file `pytest` does not satisfy it), and re-run it after any further edit. Iterate with the fast targeted commands above, gate once at the end. For a deliberate work-in-progress commit, prefix `AAI_ALLOW_COMMIT=1 git commit …`.

Individual tools (all via `uv run`):

```sh
uv run ruff check .          # lint
uv run ruff format .         # format (line-length 100)
uv run mypy                  # files = ["aai_cli", "tests"] from pyproject; strict (disallow_untyped_defs on src)
prettier --check "aai_cli/init/templates/**/*.{js,css}"  # JS/CSS template formatting
uv run pytest -q             # default unit suite
uv run pytest tests/test_transcribe.py -q              # a single file
uv run pytest tests/test_transcribe.py::test_name -q   # a single test
```

The two diff-scoped tail gates are the slowest failures to discover via the full
script; after a gate run (or any pytest run with the coverage flags below) they can
be re-run alone:

```sh
uv run pytest -q -n auto --cov=aai_cli --cov-branch --cov-context=test --cov-report=xml  # refresh coverage data
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100             # patch-coverage gate
uv run python scripts/mutation_gate.py origin/main                                       # mutation gate
```

The gate is diff-scoped, so code predating it is never mutation-tested. To audit
existing code (or a whole module) against the same bar, `scripts/mutation_sweep.py`
reuses the gate's engine over *every* line of the files you name (or the whole
package). Refresh coverage first, and pass `--timeout` to that pytest step — the
default suite has no per-test timeout (it's opt-in; see `pyproject.toml`), so a
deadlocked test would wedge the run instead of failing fast:

```sh
uv run pytest -q -n auto --timeout=60 --cov=aai_cli --cov-branch --cov-context=test --cov-report=
uv run python scripts/mutation_sweep.py aai_cli/config.py   # or omit paths for the whole package
```

### Test markers

The default suite **excludes** two slow/credentialed marker sets — `pyproject.toml`'s `addopts` carries `-m "not e2e and not install"`, so a bare `pytest` matches what `check.sh` gates. An explicit command-line `-m` overrides it for the opt-in runs:

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY, else skips
uv run pytest -m install         # installs each init template's requirements for real; needs network + uv
```

`check.sh` runs the default suite with a **90% branch-coverage gate** (`--cov-fail-under=90`). New code generally needs tests to clear that gate.

CLI output is pinned by **syrupy snapshot tests** (`tests/__snapshots__/*.ambr`). Changing help text, tables, or rendered output will fail those tests until you regenerate them with `uv run pytest --snapshot-update` and commit the updated `.ambr` files. The auto-format hook only touches `*.py`, and pre-commit's whitespace fixers deliberately skip `tests/__snapshots__/` (syrupy's indentation must stay byte-for-byte), so never hand-edit a snapshot — always regenerate.

The post-edit hook (`.claude/settings.json`) runs `ruff check --fix --unfixable F401` + `ruff format` on every edited `*.py`. `--unfixable F401` means a just-added import is **not** auto-deleted while it's momentarily unused — so adding an import in one edit and its usage in the next is safe. The flip side: a genuinely unused import survives the hook and only fails at `ruff check` in the gate, so still prefer making the import and its first usage land in the same edit.

The suite is hermetic by construction, enforced three ways (`tests/conftest.py` + `pyproject.toml` `[tool.pytest.ini_options]`): **pytest-randomly** shuffles order, an autouse `pin_timezone` fixture pins `TZ` to a fixed non-UTC zone (UTC-normalized rendering must be unaffected; use **time-machine** to freeze `now`), and **pytest-socket** (`--disable-socket`) blocks real network so an unmocked SDK/HTTP call fails loudly instead of hitting the API. A test that only binds a loopback server opts back in with the tight `@pytest.mark.allow_hosts(["127.0.0.1"])` (still blocks external hosts). The `e2e`/`install` marker suites legitimately reach the real network in-process (PyPI reachability probes, real-API runs), so a `pytest_collection_modifyitems` hook in `conftest.py` auto-grants them full sockets — adding a network marker is all that's needed, no per-test `enable_socket`.

### Writing tests that pass the diff gates

Lessons that cost iterations getting the patch-coverage and mutation tail gates green:

- **A boolean literal/default survives the mutation gate unless a test asserts the
  difference between its two values**, not just that the line ran. `json_mode=False` passed
  to `output.emit`, or `quiet=False` on `output.status`, get mutated to `True` — kill them by
  asserting the *behavioral* split: the human branch prints bare text
  (`result.output.strip() == "…"`, not a JSON object), or the spinner is actually entered
  (monkeypatch `error_console.status` and assert it ran). A changed message / `prompter.note`
  string is mutated whole, so one substring assert on the actionable keyword kills it.
- **Help text and docstrings are pinned by the syrupy snapshots, not unit asserts** — a
  mutated help string is killed by the regenerated `.ambr`, so `--snapshot-update` and commit
  rather than adding redundant `--help` substring asserts.
- **Typer's `CliRunner` merges stderr into `result.output`, and not in call order**, so don't
  assume `splitlines()[-1]` is the command payload. In `--json` mode the env-mismatch warning
  is its own `{"warning": …}` line, so filter parsed lines by a key the payload carries
  (`next(o for o in objs if "env" in o)`). A monkeypatched fake must also mirror the real
  signature — when a helper gains a kwarg (e.g. `output.status(…, quiet=…)`), doubles that
  patch it must accept it or the call `TypeError`s.
- **`--json` / `-j` is a per-command flag, not a root flag**: `assembly --json transcribe …` fails
  with "No such option"; it's `assembly transcribe … --json`. (The root callback still sniffs the
  whole token list via `argscan.requests_json`, so a callback-level failure like a bad
  `--env` keeps the JSON error shape — but the flag itself lives on the subcommand.)

### Manual QA / running the CLI in sandboxed sessions

Lessons that cost time in agent sessions — read before exercising `uv run assembly` by hand:

- **Web/remote containers are fully provisioned at session start**
  (`.claude/hooks/session-start.sh`): system deps, `markdownlint`/`prettier`, and the Go
  gate binaries (`actionlint`, `gitleaks`) are installed at CI's pinned versions, so
  `./scripts/check.sh` enforces the same gates CI does — a gate that "self-skips locally"
  should *not* be skipping in a web session. If one is, read `/tmp/session-start.log` to
  see what failed to provision. Keep the hook's stdout terse (one line per step) — it is
  injected into the agent's context every session.
- **Probe network reachability first.** Remote/sandboxed environments often allowlist
  PyPI but block `api.assemblyai.com` / `streaming.assemblyai.com` / `llm-gateway.assemblyai.com`
  (`curl -s https://api.assemblyai.com/v2/transcript -H "authorization: $ASSEMBLYAI_API_KEY"`
  returning a proxy 403 like "Host not in allowlist" means **no** real-API path can work —
  test error handling and `--show-code` instead of burning time on happy paths).
- **Isolate the config dir per test run.** The CLI persists profiles in
  `platformdirs`-resolved `config.toml` (e.g. `~/.config/assemblyai/`). Concurrent or
  destructive manual tests (corrupt-config probes, profile/env switches) stomp each other
  through that shared file — set `XDG_CONFIG_HOME=$(mktemp -d)` per run instead.
- **Write scratch output to `/tmp`, never the repo root.** Redirects like `cmd > out.txt`
  in the repo show up as untracked files and trip commit hooks/gates.
- **Headless boxes have no mic/speakers/browser.** `assembly stream`/`assembly agent` mic paths and
  `assembly login`'s browser flow can't complete; wrap exploratory runs in `timeout 30 …` so a
  blocking path can't wedge the session. For pytest, `--timeout N` (pytest-timeout, in the
  dev group) does the same per-test.

### Replay fixtures (offline end-to-end coverage)

`tests/test_replay_e2e.py` drives whole commands (`transcribe`/`transcripts`/`llm`/
`balance`/`usage`/`limits`) against **real** API responses recorded once and replayed
offline — the command's own parsing/rendering runs, but pytest-socket stays armed, so
these live in the default suite. Three moving parts:

- **`tests/fixtures/api/*.json`** — scrubbed snapshots (API key/JWT redacted, `email` and
  `account_id` faked, private `cdn.assemblyai.com/upload/…` URLs redacted). Committed and
  gitleaks-clean; treat them like syrupy snapshots (regenerate, don't hand-edit).
- **`scripts/record_fixtures.py`** — the recorder. It is **deliberately outside the gate**
  (it hits the network) and is *not* mypy/pyright-checked (only ruff covers `scripts/`).
  Refresh after an API shape change: `ASSEMBLYAI_API_KEY=… uv run python scripts/record_fixtures.py`.
  The key comes from the env; the AMS session JWT + `account_id` from the keyring/`config.toml`
  of whoever ran `assembly login` (profile `default`) — neither is ever written to a fixture.
- **`tests/replay_fixtures.py`** — rebuilds the boundary objects from JSON. A transcript is a
  real `aai.Transcript` via `Transcript.from_response`; an LLM response is rebuilt with
  `ChatCompletion.model_construct` (**not** `model_validate`) because the gateway returns
  Anthropic-flavored fields — `finish_reason="end_turn"`, token counts under
  `input_tokens`/`output_tokens` — that strict validation rejects but the OpenAI SDK itself
  parses leniently.

The replay tests patch the same boundary the unit tests do
(`commands.<cmd>.client.<fn>` / `.ams.<fn>` / `.gateway.complete`); the only difference is
the return value comes from a recorded payload instead of a hand-built mock.

## Naming & packaging gotchas

- The **package/module** is `aai_cli`; the **distribution** name is `aai-cli`; the **console command** is `assembly` (`[project.scripts] assembly = "aai_cli.main:run"`).
- `assembly init` templates live in `aai_cli/init/templates/` and are **committed**, including renamed dotfiles (`gitignore` → `.gitignore`, `env.example`). The wheel force-includes them via `[tool.hatch.build.targets.wheel] artifacts`, excluding `__pycache__/*.pyc`. Editing templates needs care — see the parametrized contract tests (`tests/test_init_template_*.py`).
- `audioop` left the stdlib in 3.13; `audioop-lts` backfills it (conditional dependency). Supported Pythons: 3.12–3.13.
- **Releasing is tag-triggered.** `.github/workflows/release.yml` fires on a pushed `vX.Y.Z` tag and builds the prebuilt arm64 Homebrew bottle (`Formula/assembly.rb`), cuts the GitHub Release, and opens the formula PR — bottling matters because the deps include Rust-backed sdists (`pydantic-core`, `jiter`, `cryptography`) that would otherwise compile from source on `brew install`. Two committed helpers drive it and are self-documenting (`--help`): `scripts/bump_patch.sh` rewrites the version in lock-step across `pyproject.toml` + `aai_cli/__init__.py` (run on a branch → merge the PR), then `scripts/cut_release.sh` tags + pushes. **`cut_release.sh` only runs from a clean `main` in sync with `origin/main`** (it hard-errors on a feature branch / dirty tree / version mismatch), so cut releases from `main`, not your working branch. The "update available" notice users see is `aai_cli/update_check.py`.

## Architecture

A Typer CLI. `aai_cli/main.py` builds the `app`, registers each command sub-app, and controls `assembly --help` ordering via `_COMMAND_ORDER` + a custom `_OrderedGroup`. `run()` is the entry point and swallows `BrokenPipeError` (closed downstream pipe → exit 0).

### Command layer

Each file in `aai_cli/commands/` is a Typer sub-app (`transcribe`, `stream`, `agent`, `speak`, `llm`, `transcripts`, `login` (login/logout/whoami), `doctor`, `init`, `dev`, `share`, `deploy`, `setup`, `onboard`, `account` (balance/usage/limits), `keys`, `sessions`, `audit`, `telemetry` (status/enable/disable)). Command bodies run through `context.run_command(ctx, fn, json=...)`, which maps any `CLIError` to clean stderr output + the error's exit code. Commands never print tracebacks for expected failures.

**Options/run split for flag-heavy commands** (gh-CLI style): the Typer function only parses argv into a frozen `<Cmd>Options` dataclass and hands it to a module-level `run_<cmd>(opts, state, *, json_mode)` through a thin lambda adapter in `run_command(ctx, ..., json=...)`. The five run commands follow it — `aai_cli/stream_exec.py` (the reference implementation), `transcribe_exec.py`, `agent_exec.py`, `speak_exec.py`, `llm_exec.py`. Because the run path is a plain function of data, tests construct options directly (`dataclasses.replace` off a defaults instance, see `tests/test_stream_exec.py` and `tests/test_command_options_seam.py`) instead of round-tripping argv through `CliRunner` — which is also the cheap way to kill mutation-gate mutants on orchestration lines. Follow this for new or heavily-reworked commands with long bodies; small commands keep the inline `body()` closure — the dataclass is pure ceremony there.

### Cross-cutting state (resolution order matters)

- **`context.py`** — `AppState` (profile, env) is attached to the Typer context in the root `@app.callback()`. `run_command` is the standard command wrapper.
- **`config.py`** — profiles persisted in `config.toml` (via `platformdirs`); the **API key lives only in the OS keyring** (`KEYRING_SERVICE = "assemblyai-cli"`), never in a dotfile. Key resolution order: `--api-key` flag (validation paths only) → `ASSEMBLYAI_API_KEY` env → keyring. **Run commands deliberately expose no `--api-key` flag** so keys can't leak into `ps`/shell history.
- **`environments.py`** — a frozen `Environment` (api_base, streaming_host, llm_gateway_base, ams_base, stytch_*). `DEFAULT_ENV` is **`production`**; use `--sandbox` (or `--env sandbox000` / `AAI_ENV`) to target the sandbox. The active environment is a process-global set once at startup; precedence: `--env` → `AAI_ENV` → profile's stored env → default. A credential is only valid against the environment that minted it.
- **`client.py`** — thin wrappers over the `assemblyai` SDK (`transcribe`, `list_transcripts`, `stream_audio`, etc.). It normalizes SDK exceptions: auth failures become a single clean `auth_failure()` `CLIError`; everything else becomes `APIError`. New SDK calls should follow this try/except shape.
- **`errors.py`** — the `CLIError` hierarchy (each with `error_type` + `exit_code`). `output.py` emits errors to **stderr**; stdout stays clean for pipelines. `--json` switches to machine-readable output; it is never auto-enabled — `output.resolve_json()` deliberately keeps human text the default even when piped or agent-run.

### Feature subsystems

- **`streaming/`** + `client.stream_audio` — v3 realtime API. Event callbacks run on the SDK reader thread and guard against `BrokenPipeError` (`stdio.silence_stdout()`) so a closed pipe never dumps a thread traceback.
- **`agent/`** — full-duplex voice agent (mic in, TTS out via `voices.py`).
- **`tts/`** + `commands/speak.py` — `assembly speak` synthesizes text to speech over the sandbox streaming-TTS WebSocket (`streaming-tts.sandbox000.…`). **Sandbox-only:** `session.is_available()` is false in production (empty `Environment.streaming_tts_host`), so the command exits 2 with a `--sandbox` hint. `session.synthesize` drives a Begin→Generate→Flush→Audio→Terminate protocol with an injectable `connect` for hermetic tests (mirrors `agent/session.py`); `audio.py` plays the PCM (default) or writes a WAV (`--out`).
- **`code_gen/`** — backs `--show-code` on `transcribe`/`stream`/`agent`: builds a ready-to-run Python SDK script from exactly the flags passed (no API key needed; generated code reads `ASSEMBLYAI_API_KEY`).
- **`auth/`** — browser-assisted `assembly login` via AMS + **Stytch B2B OAuth discovery** (`discovery.py`, `flow.py`, `loopback.py`, `ams.py`). Not Stytch Connected Apps.
- **`init/`** — scaffolds a self-contained FastAPI + HTML starter (`audio-transcription`/`live-captions`/`voice-agent` templates), optionally installs deps and opens the browser; writes the key to a git-ignored `.env`.
- **`telemetry.py`** — anonymous, opt-out usage telemetry (Supabase-CLI model): `context.run_command` wraps each command body in `telemetry.track(ctx.command_path)`, which dispatches one allow-listed event (command path, outcome/exit code, duration, version/OS, and on failure the error message capped at 500 chars — never args or account data) to the Datadog logs intake via a **detached flusher subprocess** (the hidden `assembly telemetry flush`), so commands never wait on telemetry. `SHIPPED_CLIENT_TOKEN` is a committed write-only Datadog *client* token (`pub…`, embeddable by design — never an API key; `AAI_TELEMETRY_CLIENT_TOKEN` overrides). The test suite blanks it via an autouse conftest fixture so no test ever spawns a real flusher. Opt-out: `AAI_TELEMETRY_DISABLED=1` / `DO_NOT_TRACK=1` / `assembly telemetry disable` (persisted as `telemetry_enabled` in config.toml, alongside the random `device_id`). Send-side failures are swallowed (`OSError`/`CLIError`) — telemetry must never break a command.
- **`commands/setup.py`** — `assembly setup install/status/remove` wires a coding agent up to AssemblyAI by installing three artifacts: the `assemblyai-docs` docs MCP (via `claude mcp add`), the AssemblyAI skill (via `npx skills add`), and the bundled `aai-cli` skill (copied out of the wheel, no network). Missing `claude`/`npx` is reported and skipped, not an error. The presence probes (docs MCP registered, skills on disk) live in `aai_cli/coding_agent.py` so `assembly doctor`'s coding-agent check can share them — command modules are import-linter-independent, so neither command may import the other.

## Conventions

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Ruff lint set: `E,F,I,UP,B,BLE,C4,SIM,RET,PTH,ARG,S,RUF`. `S603/S607` are ignored project-wide because the CLI intentionally shells out to `claude`/`npx` with controlled args. `B008` is ignored (Typer uses `typer.Option/Argument` calls as defaults).
- mypy is strict on `aai_cli` (`disallow_untyped_defs`); tests are type-checked but exempt from return annotations.
- Errors → stderr, data → stdout. Preserve this split; it's what makes the CLI pipeline-safe.
