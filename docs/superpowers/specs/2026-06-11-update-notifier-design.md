# Update notifier — "a new version is available" notice in the CLI

**Date:** 2026-06-11
**Status:** Approved design

## Summary

Add an unobtrusive "update available" notice to the CLI, in the style of npm's
`update-notifier` / Vercel / gh / Gemini: when a newer release exists, the next
human, interactive run prints a small bordered box on **stderr** after the command
finishes, telling the user how to upgrade. The check is **best-effort, cached, and
never blocks** — the network fetch happens in a detached background process (the
same pattern `telemetry.py` already uses), and the rendered notice always comes
from cache, so it adds zero latency to any command.

```
  ╭────────────────────────────────────────────────╮
  │                                                  │
  │   Update available  0.1.0 → 0.2.0                │
  │   Run  brew upgrade assembly  to update          │
  │                                                  │
  ╰────────────────────────────────────────────────╯
```

The upgrade command shown is **detected from how the CLI was installed** (Homebrew
/ pipx / uv tool), so the user sees the command that actually applies to them.

## Why these choices (context)

- **Version source = GitHub Releases API.** The PyPI name `assemblyai-cli` is
  squatted, so there is no PyPI to query. Releases ship as git tags + GitHub
  Releases (see the bottle pipeline), so "latest" comes from
  `GET https://api.github.com/repos/AssemblyAI/cli/releases/latest` → `tag_name`
  (e.g. `v0.2.0`). `releases/latest` excludes drafts and pre-releases, so we only
  ever notify about real, stable releases.
- **Render-from-cache + detached refresh.** This is the npm `update-notifier`
  model. The current run never waits on `api.github.com` (frequently blocked in
  sandboxes anyway); it renders whatever the last background check cached. The
  tradeoff — the first time you're behind, the box appears on the *following* run —
  is the standard, accepted behavior.
- **Reuse existing patterns.** The repo already has the exact scaffolding: the
  detached hidden-subcommand spawner in `telemetry.py`
  (`subprocess.Popen([sys.executable, "-m", "aai_cli", …], stdout=DEVNULL,
  start_new_session=True)`), the per-command hook in `context.run_command`
  (where `telemetry.track` is invoked), the stderr/TTY/JSON discipline in
  `output.py`, and non-secret persisted state in `config.toml`.

## Architecture

One new module, `aai_cli/update_check.py`, plus a hidden `_update-check`
subcommand for the network fetch, hooked into `context.run_command`. Mirrors the
`telemetry.py` + hidden `telemetry flush` split.

### Data flow

1. A command runs. After the body completes, `run_command` calls
   `update_check.maybe_notify(ctx.command_path)`.
2. `maybe_notify` checks the gates (below). If they pass and `config.toml`'s cached
   `update_latest_version` is newer than `aai_cli.__version__`, it renders the box
   on `output.error_console`.
3. Independently, if the cache is **stale** (`now - update_last_check > 24h`, or
   never checked), `maybe_notify` calls `spawn_refresh()` — a detached
   `assembly _update-check` process — and returns immediately. That process GETs
   the releases API, parses `tag_name`, and writes `update_latest_version` +
   `update_last_check` into `config.toml`. The result is used by the *next* run.

So the only synchronous work on the hot path is: read two config fields, compare
two versions, maybe print a panel, maybe fork a detached process. No network, no
blocking.

### Components (`aai_cli/update_check.py`)

- `maybe_notify(command_path: str) -> None` — the single entry point
  `run_command` calls. Orchestrates gating → render-from-cache → maybe spawn
  refresh. Swallows everything.
- `_should_notify() -> bool` — the gate (see "Suppression").
- `detect_upgrade_command() -> str` — inspect `sys.executable` (the venv/bin the
  running `assembly` lives in):
  - path under a Homebrew prefix (`/opt/homebrew`, `/usr/local`, or
    `$(brew --prefix)` Cellar) → `brew upgrade assembly`
  - path under a pipx venv (contains `pipx/venvs`, or under `$PIPX_HOME`) →
    `pipx upgrade assembly`
  - path under a uv tools dir (contains `uv/tools`, or under `$UV_TOOL_DIR`) →
    `uv tool upgrade assembly`
  - otherwise → a generic `See https://github.com/AssemblyAI/cli#installation`
    hint.
- `_render(current: str, latest: str, upgrade: str) -> None` — build a Rich
  `Panel` and print it to `output.error_console`. Theme-consistent with the rest
  of the CLI (reuse `aai_cli/theme.py` styles).
- `spawn_refresh() -> None` — detached `Popen` clone of telemetry's spawner;
  `start_new_session=True`, stdio → `DEVNULL`. Guarded so the `_update-check`
  process can never spawn another (mirrors telemetry's self-spawn guard).
- `fetch_and_cache() -> None` — the hidden subcommand body. stdlib
  `urllib.request` GET with a `User-Agent` header (GitHub requires one), 5s
  timeout, parse `tag_name`, normalize (`lstrip("v")`), write the two config
  fields. **Every exception swallowed** (`URLError`, `OSError`, JSON/key errors,
  rate-limit non-200s).
- `is_newer(latest: str, current: str) -> bool` — `packaging.version.Version`
  compare; `InvalidVersion` → `False` (never notify on a version we can't parse).

### Hidden subcommand

Register `_update-check` (`hidden=True`) the same way `telemetry flush` is
registered — an explicit, reviewable entry point invoked as
`python -m aai_cli _update-check`. Its body is `fetch_and_cache()`.

### `config.py` additions

Two new optional fields on the `Config` dataclass, persisted in `config.toml`
alongside `telemetry_enabled`/`device_id`:

- `update_last_check: float | None` — unix timestamp of the last fetch attempt.
- `update_latest_version: str | None` — the last `tag_name` seen (without the
  `v`).

Plus getters/setters following the existing `get_telemetry_enabled` /
`set_telemetry_enabled` shape.

### Dependency

Declare `packaging` as a direct `[project.dependencies]` entry (it's already
resolved transitively at 26.2 in `uv.lock`, so it pins to the same version — no
new download). Use a **conservative floor** (e.g. `packaging>=24.0`) per the
project's safe-chain age-gate behavior, then `uv lock` so `uv lock --check`
passes. This replaces hand-rolled version parsing with the standard, robust
`Version` comparator and keeps `deptry` happy (no transitive dep used directly).

## Suppression (no notice when …)

`_should_notify()` returns `False` — and no box prints — when **any** of:

- the active output mode is JSON / agentic (`output.resolve_json(...)` true, or
  the agentic signal in `output.py`),
- **stderr is not a TTY** (piped or redirected) — the box would corrupt captured
  stderr,
- `AAI_NO_UPDATE_CHECK` is set (any non-empty value),
- `CI` is set (don't nag in pipelines),
- the command is the hidden `_update-check` itself (and the `telemetry flush`
  hidden command),
- there is no cached version, or the cached version is not newer.

The notice prints **only on a command's success path** — not stacked under an
error message. (Considered and rejected: always-show, à la Vercel — an update box
directly under an error reads as noise.)

## Error handling

Best-effort throughout, matching `telemetry.py`: the synchronous path catches and
swallows config-read and render errors; the detached `fetch_and_cache` swallows
all network/parse/IO errors. A failure anywhere means "no notice this run," never
a traceback and never a delayed or broken command.

## Testing

Hermetic (pytest-socket stays armed; the fetch is mocked, the spawn is
monkeypatched), mirroring the telemetry tests:

- `detect_upgrade_command`: monkeypatch `sys.executable` to a brew-prefix path, a
  `pipx/venvs` path, a `uv/tools` path, and an unknown path → asserts each returns
  the right command / generic hint.
- `is_newer`: newer, equal, older, and an unparseable string → `False`.
- `maybe_notify` gating (behavioral asserts that survive the mutation gate):
  - newer version cached + TTY + human → the box prints, and its text contains
    the `current → latest` arrow and the detected command (substring assert on the
    actionable keyword).
  - **no** output under `--json`, when stderr is not a TTY, when `CI` is set, and
    when `AAI_NO_UPDATE_CHECK` is set (assert empty stderr for each).
  - cache equal/older → no box.
- `maybe_notify` spawns the refresh only when the cache is stale: monkeypatch
  `Popen`, freeze time with **time-machine**, assert the detached args +
  `start_new_session=True` when stale, and **no** spawn when fresh.
- `fetch_and_cache`: feed a sample `releases/latest` JSON via a mocked opener →
  asserts the two config fields are written and `v`-prefix stripped; feed an error
  / non-200 / bad JSON → asserts it swallows and writes nothing (or only the
  timestamp).
- syrupy snapshot of the rendered panel (so its exact look is pinned, like other
  CLI output).

## Out of scope

- A `assembly update`/self-update command that performs the upgrade (we only
  *notify*; upgrading is the package manager's job).
- A config-file UI or `assembly` subcommand to toggle the check — `AAI_NO_UPDATE_CHECK`
  + `CI` detection is enough (YAGNI).
- Checking more often than daily, or any synchronous/blocking check.
- Notifying about pre-releases (the releases API excludes them).
- Folding the refresh into the telemetry flusher — kept separate so update checks
  and telemetry opt-out independently (the extra process only spawns ~once/day).
