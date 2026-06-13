# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, and
others) when working with code in this repository. `CLAUDE.md` is a symlink to
this file, so Claude Code reads the same instructions.

**Guidance is split per directory** so many agents can update it concurrently
without conflicting in one file. This root file holds repo-wide invariants;
read the `AGENTS.md` nearest the code you're changing:

- `aai_cli/AGENTS.md` — architecture, the command-registration convention,
  cross-cutting state, feature subsystems.
- `tests/AGENTS.md` — test markers, snapshot goldens, hermeticity rules, and
  the hard-won lessons for getting the patch-coverage and mutation gates green.

## Development commands

This project uses [uv](https://docs.astral.sh/uv/). **Run every Python tool through `uv run`** so it uses the locked environment (`pyproject.toml` + `uv.lock`), not whatever is on `PATH`:

```sh
uv sync                      # create/refresh the venv (the dev group installs by default)
uv run assembly --help            # run the CLI from the locked environment
./scripts/check.sh           # the full gate CI runs (scripts/check.sh is the source of truth)
```

Dev tooling is a PEP 735 `[dependency-groups]` group with `default-groups = ["dev"]`, not a `[project]` extra — `uv sync --extra dev` errors.

`scripts/check.sh` is the authoritative gate; keep this list in sync with it. It runs, in order: `uv lock --check` → `ruff check` → `ruff format --check` → `mypy` → `pyright` (src strict) → `pyright` (tests) → `vulture` (dead code) → `deptry` (dependency hygiene) → `lint-imports` (import-linter architecture contracts) → max-file-length (500 lines) → `xenon` (cyclomatic complexity, max grade B / project avg A) → `swiftlint` + swift compile (macOS only, skipped elsewhere) → `markdownlint` → `codespell` (spell-check code/comments/docs via `uvx`; config in `[tool.codespell]`) → `prettier` (init template JS/CSS) → `shellcheck` → `actionlint` + `zizmor` (workflow lint/audit) → `gitleaks` (secret scan) → generated `--show-code` compile gate → init template contract gate → unused snapshot/fixture gate (`scripts/unused_fixtures_gate.py`: orphaned `.ambr`/API fixtures, since xdist disables syrupy's own unused detection) → docs consistency gate (`scripts/docs_consistency_gate.py`: REFERENCE.md/README.md env vars, exit codes, and `assembly …` command refs stay in sync with the code) → docstring coverage gate (`scripts/docstring_coverage_gate.py`: public-API docstring ratchet, an `interrogate` stand-in that handles PEP 695 generics) → `brew audit --strict` (the shipped `Formula/assembly.rb`; self-skips without Homebrew) → `pytest` (90% branch coverage) → `diff-cover` (100% patch coverage vs `origin/main`) → **mutation gate** (diff-scoped: mutates each changed line and reruns the tests that cover it — a surviving mutant fails the gate, so changed lines need assertions that would *fail* if the line broke, not just coverage; suppress a genuinely unassertable line with `# pragma: no mutate`) → a "no new escape hatches" gate (`# type: ignore` / `# noqa` / `pragma: no cover` / `Any` / `cast(` / test skip/xfail/sleep, all **count-gated against the merge-base** so moving an existing hatch in a refactor doesn't false-positive but a net-new one fails) → **CodeQL gate** (`scripts/codeql_gate.py`: the same security + quality suites the CodeQL workflow uploads to GitHub's code-scanning/quality tabs, run locally over python/actions/javascript so alerts fail before push instead of on the PR; needs the CodeQL bundle on PATH — self-skips otherwise, `codeql.yml` covers CI, and the web session-start hook provisions it) → `uv build` + `twine check --strict`. The `vulture`/`deptry`/`lint-imports`/`xenon`, patch-coverage, and mutation stages catch the failures that `ruff`+`mypy` alone won't — don't claim the gate is green until the script prints `All checks passed.`

**Commits are gated.** On success `check.sh` records a working-tree signature (`scripts/gate_marker.py record` → `.git/aai-gate-pass`), and a PreToolUse hook (`.claude/hooks/require-gate-before-commit.sh`) blocks `git commit` unless that signature still matches — so run the full gate to completion *before* committing (a single-file `pytest` does not satisfy it), and re-run it after any further edit. Iterate with the fast targeted commands above, gate once at the end. For a deliberate work-in-progress commit, prefix `AAI_ALLOW_COMMIT=1 git commit …`.

Individual tools (all via `uv run`):

```sh
uv run ruff check .          # lint
uv run ruff format .         # format (line-length 100)
uv run mypy                  # files = ["aai_cli", "tests"] from pyproject; src is full --strict bar disallow_untyped_calls (jiwer ships no stubs); tests relax the untyped-body flags
prettier --check "aai_cli/init/templates/**/*.{js,css}"  # JS/CSS template formatting
uv run pytest -q             # default unit suite
uv run pytest tests/test_transcribe.py -q              # a single file
uv run pytest tests/test_transcribe.py::test_name -q   # a single test
```

The post-edit hook (`.claude/settings.json`) runs `ruff check --fix --unfixable F401` + `ruff format` on every edited `*.py`. `--unfixable F401` means a just-added import is **not** auto-deleted while it's momentarily unused — so adding an import in one edit and its usage in the next is safe. The flip side: a genuinely unused import survives the hook and only fails at `ruff check` in the gate, so still prefer making the import and its first usage land in the same edit.

## Working alongside other agents

Dozens of sessions may be working on this repo concurrently; the codebase is
structured so independent changes stay in disjoint files. Keep it that way:

- **Check for in-flight duplicates before starting a fix.** Before implementing
  a bug fix or small feature, scan open PRs and the last few `origin/main`
  commits touching the same files (two sessions once shipped the identical fix;
  the slower PR was closed as redundant). The `pr-overlap` workflow also warns
  when a PR's changed files intersect another open PR's — treat that warning as
  a prompt to reconcile, not noise.
- **A new command edits no shared file.** Registration, help ordering, and the
  snapshot partition are all derived from the command module's own `SPEC`
  declaration (see `aai_cli/AGENTS.md`). If you find yourself editing a shared
  list to add a command, you're fighting the convention.
- **Dependency changes are not part of feature PRs.** `uv.lock` is the one file
  two branches can never merge cleanly; add or bump dependencies in a
  dedicated, single-purpose PR so feature branches don't collide in the
  lockfile.
- **Land through the merge queue.** The diff-scoped gates compare against
  `origin/main`, which moves constantly; two individually-green PRs can be
  jointly red. PRs should merge via GitHub's merge queue (a repository setting)
  so the gate re-runs against the combined state before landing — don't bypass
  it with direct pushes to `main`.
- **Update the `AGENTS.md` nearest your change** when you learn something
  durable; don't grow this root file.

## Naming & packaging gotchas

- The **package/module** is `aai_cli`; the **distribution** name is `aai-cli`; the **console command** is `assembly` (`[project.scripts] assembly = "aai_cli.main:run"`).
- `assembly init` templates live in `aai_cli/init/templates/` and are **committed**, including renamed dotfiles (`gitignore` → `.gitignore`, `env.example`). The wheel force-includes them via `[tool.hatch.build.targets.wheel] artifacts`, excluding `__pycache__/*.pyc`. Editing templates needs care — see the parametrized contract tests (`tests/test_init_template_*.py`).
- `audioop` left the stdlib in 3.13; `audioop-lts` backfills it (conditional dependency). Supported Pythons: 3.12–3.13.
- **Releasing is tag-triggered.** The version is **derived from the git tag** by hatch-vcs and written to a gitignored `aai_cli/_version.py` at build time — there is no version string to keep in sync across `pyproject.toml` or `aai_cli/__init__.py`, and `bump_patch.sh` no longer exists. To cut a release, run `scripts/cut_release.sh` from a clean `main` in sync with `origin/main`: no argument → next patch above the latest `vX.Y.Z` tag; `cut_release.sh X.Y.Z` → explicit version. It tags + pushes, which fires `.github/workflows/release.yml` — that builds the prebuilt arm64 Homebrew bottle (`Formula/assembly.rb`), cuts the GitHub Release, and opens the formula PR. **You don't need a local checkout to release:** `release.yml` also has a manual `workflow_dispatch` (GitHub's "Run workflow" button, or `actions_run_trigger` from a Claude web session) taking an optional `version` input — its `tag` job resolves the version and creates+pushes the tag (reusing `cut_release.sh --no-push`), and the rest of the pipeline then runs in that same workflow run. Tag creation lives *inside* the release run on purpose: a `GITHUB_TOKEN` tag push wouldn't re-trigger the `on: push` half, so a separate "push the tag" workflow would silently never build. (`dry_run: true` builds the bottle for an existing tag without publishing.) Bottling matters because the deps include Rust-backed sdists (`pydantic-core`, `jiter`, `cryptography`) that would otherwise compile from source on `brew install`. The Homebrew formula builds from a git-less GitHub source tarball, so `Formula/assembly.rb`'s `def install` sets the generic `SETUPTOOLS_SCM_PRETEND_VERSION` env var (installing resources first under a clean env, then setting the var for our package only) to feed the tag version to the build. **`cut_release.sh` only runs from a clean `main` in sync with `origin/main`** (it hard-errors on a feature branch / dirty tree), so cut releases from `main`, not your working branch. The "update available" notice users see is `aai_cli/update_check.py`.

## Manual QA / running the CLI in sandboxed sessions

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

## Conventions

- `from __future__ import annotations` at the top of every module; modern typing (`X | None`).
- Ruff lint set: see `[tool.ruff.lint]` in `pyproject.toml`. `S603/S607` are ignored project-wide because the CLI intentionally shells out to `claude`/`npx` with controlled args. `B008` is ignored (Typer uses `typer.Option/Argument` calls as defaults).
- mypy is strict on `aai_cli` (`disallow_untyped_defs`); tests are type-checked but exempt from return annotations.
- Errors → stderr, data → stdout. Preserve this split; it's what makes the CLI pipeline-safe.
- **Help copy is terse and period-less (Codex-CLI style)**: one-line command summaries (the docstring's first line) and single-sentence option/argument `help=` strings are imperative, sentence-case, and carry **no trailing period** — `"Burn always-visible captions into a video"`, not `"…video."`. Only genuinely multi-sentence help (e.g. `"X. Default: Y."`) keeps normal punctuation. The strings render in `assembly --help`, so they're pinned by the syrupy `--help` goldens (`tests/__snapshots__/test_snapshots_help_*.ambr`) — regenerate with `--snapshot-update`, never hand-edit. Don't drop the period on internal helper docstrings (they aren't snapshot-covered, so the mutation gate would flag the changed line).
- **Deprecate flags with hidden traps, not removal**: keep the old flag parsing (`hidden=True`), emit a one-line "use X instead" warning, and drop it a release or two later — never hard-break a script mid-cycle. `login --api-key` (→ `--with-api-key`) is the pattern to copy.
- **Secrets never ride argv**: a key/token-valued option must read from stdin (`--with-api-key`) or the env, so it can't leak into shell history or `ps`. Run commands deliberately have no `--api-key` at all.
- **Every NDJSON stream line carries a `"type"` field** (see REFERENCE.md "JSON output"); new event types are additive, existing fields stay stable.
