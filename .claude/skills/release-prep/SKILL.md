---
name: release-prep
description: Prepare an assembly CLI release — confirm main is green, then tag (locally or via the manual workflow) to trigger the bottle pipeline. Use when cutting a new release.
disable-model-invocation: true
---

# release-prep

Drive an `assembly` release to a verified, tagged state. Stop and report at the first failure — never tag on a red check.

## 1. Pick the version

- With hatch-vcs **the git tag _is_ the version** — there is no `pyproject.toml` / `aai_cli/__init__.py` string to bump. `cut_release.sh` defaults to the next patch above the latest `vX.Y.Z` tag; pass `X.Y.Z` for a minor/major bump.
- Decide the bump (patch/minor/major) from what changed since the last tag; ask the user if it's ambiguous.

## 2. Full gate

```sh
./scripts/check.sh
```

Must end with `All checks passed.` (ruff, mypy, markdownlint, shellcheck, pytest+coverage, build, `twine check --strict`). The release builds whatever `main` points at, so confirm `main` is green before tagging.

## 3. Tag to trigger the bottle pipeline

Two equivalent ways to cut the tag — both land on `.github/workflows/release.yml`:

**Local** (from a clean `main` in sync with `origin/main`):

```sh
./scripts/cut_release.sh           # next patch; --dry-run verifies without tagging, --yes skips the prompt
./scripts/cut_release.sh 0.3.0     # explicit version
```

**No local checkout** (e.g. a Claude web session on a feature branch): run the **Release** workflow's manual `workflow_dispatch` — GitHub's "Run workflow" button, or the `actions_run_trigger` MCP tool — with an optional `version` input (blank = next patch). Its `tag` job resolves the version and creates+pushes the tag from `main`, then the same run builds and publishes. Set `dry_run: true` to build the bottle for an existing tag without publishing.

The tag triggers `.github/workflows/release.yml`, which:

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
