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

   This runs, in order: `ruff check` → `ruff format --check` → `mypy` (src + tests) → `markdownlint` (excludes generated `docs/`) → `shellcheck install.sh` → `pytest` with a **90% branch-coverage gate** (`--cov-fail-under=90`, excluding `e2e` and `install_script` markers) → `uv build` + `twine check --strict`. Everything runs through `uv run` against the locked environment.

2. If anything fails, fix it and re-run `./scripts/check.sh` until it passes. Do not claim success until the script prints `All checks passed.`

## Optional, opt-in suites (not run by check.sh)

Run these only when relevant — they are slow and/or need credentials:

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY + kokoro
uv run pytest -m install_script  # builds a wheel and runs install.sh for real; needs network + uv/pipx
```

## Notes

- If `shellcheck` isn't installed locally, `check.sh` skips it with a notice (CI still runs it) — that's expected, not a failure.
- Report the final outcome with the actual tail of the output, not a summary from memory.
