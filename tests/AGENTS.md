# tests/ — test-suite guide

Scoped guidance for the test suite. Repo-wide invariants (gate, commit hooks)
live in the root `AGENTS.md`; architecture lives in `aai_cli/AGENTS.md`.

## Test markers

The default suite **excludes** two slow/credentialed marker sets — `pyproject.toml`'s `addopts` carries `-m "not e2e and not install"`, so a bare `pytest` matches what `check.sh` gates. An explicit command-line `-m` overrides it for the opt-in runs:

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY, else skips
uv run pytest -m install         # installs each init template's requirements for real; needs network + uv
```

`check.sh` runs the default suite with a **90% branch-coverage gate** (`--cov-fail-under=90`). New code generally needs tests to clear that gate.

## Snapshot goldens

CLI output is pinned by **syrupy snapshot tests** (`tests/__snapshots__/*.ambr`). Changing help text, tables, or rendered output will fail those tests until you regenerate them with `uv run pytest --snapshot-update` and commit the updated `.ambr` files. The auto-format hook only touches `*.py`, and pre-commit's whitespace fixers deliberately skip `tests/__snapshots__/` (syrupy's indentation must stay byte-for-byte), so never hand-edit a snapshot — always regenerate.

The `--help` goldens are split per command group (`tests/test_snapshots_help_<group>.py`) so concurrent branches touching different commands regenerate *different* `.ambr` files. The partition (`HELP_GROUPS` in `tests/_snapshot_surface.py`) is **derived from each command module's `SPEC.panel`** (see `aai_cli/command_registry.py`), so a new command lands in the right group automatically; `tests/test_snapshots_help_groups.py` guards that the derived partition matches the live Typer tree. The root `assembly --help` screen — which every new command changes — has its own golden (`tests/test_snapshots_help_root.py`), so that churn stays confined to one trivially-regenerable `.ambr` file.

## Hermeticity (enforced three ways)

The suite is hermetic by construction (`tests/conftest.py` + `pyproject.toml` `[tool.pytest.ini_options]`): **pytest-randomly** shuffles order, an autouse `pin_timezone` fixture pins `TZ` to a fixed non-UTC zone (UTC-normalized rendering must be unaffected; use **time-machine** to freeze `now`), and **pytest-socket** (`--disable-socket`) blocks real network so an unmocked SDK/HTTP call fails loudly instead of hitting the API. A test that only binds a loopback server opts back in with the tight `@pytest.mark.allow_hosts(["127.0.0.1"])` (still blocks external hosts). The `e2e`/`install` marker suites legitimately reach the real network in-process (PyPI reachability probes, real-API runs), so a `pytest_collection_modifyitems` hook in `conftest.py` auto-grants them full sockets — adding a network marker is all that's needed, no per-test `enable_socket`.

**Tests that touch global logging state must snapshot/restore it** — root handlers/level and per-logger levels are process-global, so a leak only fails on some pytest-randomly seeds (green locally, red in CI). Opt in to the shared `preserve_logging_state` conftest fixture (it also resets the websockets wire loggers a silencer test may have clamped) instead of hand-rolling the snapshot per module.

## Writing tests that pass the diff gates

Lessons that cost iterations getting the patch-coverage and mutation tail gates green:

- **Control a command's output shape with the real `--json` flag, never by patching
  `output.resolve_json`.** `resolve_json` is now just `return explicit` (it no longer
  auto-enables JSON off a tty), so a test wanting human output simply omits `--json`
  (the suite's default) and one wanting machine output passes `--json` to `runner.invoke`.
  A `monkeypatch.setattr("aai_cli.output.resolve_json", …)` is therefore a no-op that
  bypasses the real argscan→`json_option`→`resolve_json` path — don't add one.
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
- **Render width is pinned suite-wide (`COLUMNS=80`), so a `--help` substring assert is
  deterministic** — the autouse `fixed_render_size` fixture (`conftest.py`) sets `COLUMNS`/`LINES`
  for *every* test, because the no-clip help table ellipsizes a long flag name
  (`--end-of-turn-c…`) once its column overflows. Without the pin a `--with-api-key`-style
  substring assert passes at a wide local terminal and fails at CI's narrower width — that gap
  cost a PR three CI rounds. Don't fight it: a local green is now a CI green for output tests.
  A test that genuinely needs a different width passes it on the call
  (`runner.invoke(app, argv, env={"COLUMNS": "300"})`), which overrides the default.
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

## Replay fixtures (offline end-to-end coverage)

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
