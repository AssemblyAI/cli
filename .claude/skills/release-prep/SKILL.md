---
name: release-prep
description: Prepare an assembly CLI release — bump the version, run the full gate, then tag to trigger the bottle pipeline. Use when cutting a new release.
disable-model-invocation: true
---

# release-prep

Drive an `assembly` release to a verified, tagged state. Stop and report at the first failure — never tag on a red check.

## 1. Version bump

- Update `version` in `pyproject.toml` (`[project]`). Confirm `aai_cli/__init__.py` `__version__` stays in sync (the `version` command reads it).
- Decide the bump (patch/minor/major) from what changed since the last tag; ask the user if it's ambiguous.
- Land the bump via a normal PR (regular CI) before tagging.

## 2. Full gate

```sh
./scripts/check.sh
```

Must end with `All checks passed.` (ruff, mypy, markdownlint, shellcheck, pytest+coverage, build, `twine check --strict`).

## 3. Tag to trigger the bottle pipeline

```sh
./scripts/cut_release.sh
```

This derives the version from `pyproject.toml`, verifies the tree is clean, on `main`, and in sync with origin, then tags `vX.Y.Z` and pushes it. (`--dry-run` verifies without tagging; `--yes` skips the confirmation prompt.)

The pushed tag triggers `.github/workflows/release.yml`, which:

1. Builds the arm64 macOS bottle (`arm64_sonoma`).
2. Creates the `vX.Y.Z` GitHub Release with the bottle attached.
3. Opens a `release/vX.Y.Z-formula` PR pinning the formula to the tag's source and adding the `bottle do` block.

## 4. Merge the formula PR

Review the `release/vX.Y.Z-formula` PR (formula-only diff) and merge it with the repo-admin **"merge without waiting for requirements"** override — a PR opened by `GITHUB_TOKEN` doesn't trigger CI, so the required check won't report on its own.

## 5. Verify the bottle

On a clean arm64 Mac:

```sh
brew update && brew install assembly   # pulls the bottle — fast, no rust/llvm
assembly --version                     # matches the tagged version
```

## Distribution caveat

The PyPI name **`assemblyai-cli` is squatted by a third party** — `pip install assemblyai-cli` does **not** resolve to this project. Distribution is the **Homebrew bottle** (primary, macOS arm64) and **pipx/uv `git+https`** (fallback, all platforms). There is no PyPI publish step.
