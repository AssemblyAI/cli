---
name: check
description: Run the full local verification gate for the aai CLI (the same checks CI runs). Use before pushing or opening a PR.
disable-model-invocation: true
---

# check

Run the project's canonical verification gate and report the result.

## Steps

1. Run the full gate:

   ```sh
   ./scripts/check.sh
   ```

   This runs, in order: `uv lock --check` → `ruff check` → `ruff format --check` → `mypy` (src + tests) → `pyright` (src strict, then tests) → `vulture` (dead code) → `deptry` (dependency hygiene) → `lint-imports` (architecture contracts) → `xenon` (cyclomatic complexity, max grade B / project avg A) → `swiftlint` + swift compile (macOS only) → `markdownlint` (excludes generated `docs/`) → `prettier` (init template JS/CSS) → `shellcheck install.sh scripts/check.sh` → generated `--show-code` compile gate → init template contract gate → `pytest` with a **90% branch-coverage gate** (`--cov-fail-under=90`, excluding `e2e`/`install`/`install_script` markers) → `diff-cover` (100% patch coverage vs `origin/main`) → a "no new escape hatches" diff gate → `uv build` + `twine check --strict`. Everything Python runs through `uv run` against the locked environment.

   Heads-up on the stages `ruff`+`mypy` don't cover: `vulture` flags unused code, `deptry` flags unused/missing/misplaced dependencies, `lint-imports` enforces the import-architecture contracts in `.importlinter`, and `xenon` fails any function over cyclomatic-complexity grade B (CC > 10). These are the ones that most often surprise an otherwise-clean change.

2. If anything fails, fix it and re-run `./scripts/check.sh` until it passes. Do not claim success until the script prints `All checks passed.`

## Optional, opt-in suites (not run by check.sh)

Run these only when relevant — they are slow and/or need credentials:

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY + kokoro
uv run pytest -m install_script  # builds a wheel and runs install.sh for real; needs network + uv/pipx
```

## Notes

- External linters that aren't Python deps — `shellcheck`, `prettier`, `swiftlint`/`swiftc` — self-skip with a notice when not installed (CI still runs them); that's expected, not a failure. `swiftlint`/`swiftc` also no-op off macOS.
- `diff-cover` and the escape-hatch gate self-skip when `origin/main` isn't present (e.g. a shallow branch-only clone); CI provides the base ref.
- Report the final outcome with the actual tail of the output, not a summary from memory.
