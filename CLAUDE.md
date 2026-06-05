# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

This project uses [uv](https://docs.astral.sh/uv/). **Run every Python tool through `uv run`** so it uses the locked environment (`pyproject.toml` + `uv.lock`), not whatever is on `PATH`:

```sh
uv sync --extra dev          # create/refresh the venv with dev dependencies
uv run aai --help            # run the CLI from the locked environment
./scripts/check.sh           # the full gate CI runs: ruff + mypy + markdownlint + shellcheck + pytest(+coverage) + build/twine
```

Individual tools (all via `uv run`):

```sh
uv run ruff check .          # lint
uv run ruff format .         # format (line-length 100)
uv run mypy                  # files = ["aai_cli", "tests"] from pyproject; strict (disallow_untyped_defs on src)
uv run pytest -q             # default unit suite
uv run pytest tests/test_transcribe.py -q              # a single file
uv run pytest tests/test_transcribe.py::test_name -q   # a single test
```

### Test markers

The default suite **excludes** two slow/credentialed marker sets (see `scripts/check.sh` and `pyproject.toml`):

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY, else skips
uv run pytest -m install_script  # builds a wheel and runs install.sh for real; needs network + uv/pipx
```

`check.sh` runs `-m "not e2e and not install_script"` with a **90% branch-coverage gate** (`--cov-fail-under=90`). New code generally needs tests to clear that gate.

## Naming & packaging gotchas

- The **package/module** is `aai_cli`; the **distribution** name is `aai-cli`; the **console command** is `aai` (`[project.scripts] aai = "aai_cli.main:run"`).
- `aai init` templates live in `aai_cli/init/templates/` and are **committed**, including renamed dotfiles (`gitignore` → `.gitignore`, `env.example`). The wheel force-includes them via `[tool.hatch.build.targets.wheel] artifacts`, excluding `__pycache__/*.pyc`. Editing templates needs care — see the parametrized contract tests (`tests/test_init_template_*.py`).
- `audioop` left the stdlib in 3.13; `audioop-lts` backfills it (conditional dependency). Supported Pythons: 3.10–3.13.

## Architecture

A Typer CLI. `aai_cli/main.py` builds the `app`, registers each command sub-app, and controls `aai --help` ordering via `_COMMAND_ORDER` + a custom `_OrderedGroup`. `run()` is the entry point and swallows `BrokenPipeError` (closed downstream pipe → exit 0).

### Command layer

Each file in `aai_cli/commands/` is a Typer sub-app (`transcribe`, `stream`, `transcripts`, `agent`, `llm`, `login`, `doctor`, `samples`, `init`, `claude`). Command bodies run through `context.run_command(ctx, fn, json=...)`, which maps any `CLIError` to clean stderr output + the error's exit code. Commands never print tracebacks for expected failures.

### Cross-cutting state (resolution order matters)

- **`context.py`** — `AppState` (profile, env) is attached to the Typer context in the root `@app.callback()`. `run_command` is the standard command wrapper.
- **`config.py`** — profiles persisted in `config.toml` (via `platformdirs`); the **API key lives only in the OS keyring** (`KEYRING_SERVICE = "assemblyai-cli"`), never in a dotfile. Key resolution order: `--api-key` flag (validation paths only) → `ASSEMBLYAI_API_KEY` env → keyring. **Run commands deliberately expose no `--api-key` flag** so keys can't leak into `ps`/shell history.
- **`environments.py`** — a frozen `Environment` (api_base, streaming_host, llm_gateway_base, ams_base, stytch_*). `DEFAULT_ENV` is currently **`sandbox000`** (flip to `production` once prod AMS/Stytch values are real). The active environment is a process-global set once at startup; precedence: `--env` → `AAI_ENV` → profile's stored env → default. A credential is only valid against the environment that minted it.
- **`client.py`** — thin wrappers over the `assemblyai` SDK (`transcribe`, `list_transcripts`, `stream_audio`, etc.). It normalizes SDK exceptions: auth failures become a single clean `auth_failure()` `CLIError`; everything else becomes `APIError`. New SDK calls should follow this try/except shape.
- **`errors.py`** — the `CLIError` hierarchy (each with `error_type` + `exit_code`). `output.py` emits errors to **stderr**; stdout stays clean for pipelines. `--json` (auto-enabled when piped/agent-run) switches to machine-readable output.

### Feature subsystems

- **`streaming/`** + `client.stream_audio` — v3 realtime API. Event callbacks run on the SDK reader thread and guard against `BrokenPipeError` (`stdio.silence_stdout()`) so a closed pipe never dumps a thread traceback.
- **`agent/`** — full-duplex voice agent (mic in, TTS out via `voices.py`).
- **`code_gen/`** — backs `--show-code` on `transcribe`/`stream`/`agent`: builds a ready-to-run Python SDK script from exactly the flags passed (no API key needed; generated code reads `ASSEMBLYAI_API_KEY`).
- **`auth/`** — browser-assisted `aai login` via AMS + **Stytch B2B OAuth discovery** (`discovery.py`, `flow.py`, `loopback.py`, `ams.py`). Not Stytch Connected Apps.
- **`init/`** — scaffolds a self-contained FastAPI + HTML starter (`audio-transcription`/`live-captions`/`voice-agent` templates), optionally installs deps and opens the browser; writes the key to a git-ignored `.env`.
- **`commands/claude.py`** — `aai claude install/status/remove` shells out to `claude mcp add` (the `assemblyai-docs` MCP) and `npx skills add` (the AssemblyAI skill). Missing `claude`/`npx` is reported and skipped, not an error.

## Conventions

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Ruff lint set: `E,F,I,UP,B,BLE,C4,SIM,RET,PTH,ARG,S,RUF`. `S603/S607` are ignored project-wide because the CLI intentionally shells out to `claude`/`npx` with controlled args. `B008` is ignored (Typer uses `typer.Option/Argument` calls as defaults).
- mypy is strict on `aai_cli` (`disallow_untyped_defs`); tests are type-checked but exempt from return annotations.
- Errors → stderr, data → stdout. Preserve this split; it's what makes the CLI pipeline-safe.
