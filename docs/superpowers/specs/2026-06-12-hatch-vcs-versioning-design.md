# Tag-derived versioning via hatch-vcs

**Date:** 2026-06-12
**Status:** Approved — pending spec review

## Goal

Make releases use idiomatic hatchling: derive the package version from the git
tag via `hatch-vcs` instead of hand-syncing a literal version string across two
files. Releasing collapses to "tag `main`"; there is no version string to bump.

## Motivation

Today the version lives in **two** hand-synced places — `pyproject.toml`
(`version = "0.1.4"`) and `aai_cli/__init__.py` (`__version__ = "0.1.4"`) — kept
in lock-step by `scripts/bump_patch.sh`, which also refreshes `uv.lock`.
`scripts/cut_release.sh` then tags whatever `pyproject.toml` holds. The dual-file
sync is pure ceremony: the tag is already the real source of release truth.

## Constraints (must not break)

1. **The Homebrew bottle builds from a GitHub source tarball**
   (`archive/refs/tags/vX.Y.Z.tar.gz`) via `virtualenv_install_with_resources`.
   That tarball has **no `.git` directory**, so deriving the version from git at
   *install* time yields nothing.
2. **The formula's `test do` asserts `assembly --version` == the tag version**
   (`assert_match version.to_s, shell_output("#{bin}/assembly --version")`), so
   the installed build must report the real version even without git.
3. **The gate runs `uv lock --check`** — the project's own version is recorded in
   `uv.lock`, so a per-commit-drifting version could make the gate perpetually red.
4. **`__version__` is read at runtime in five modules** — `main.py` (`--version`
   callback), `telemetry.py` (ddtags + payload), `output.py` (banner), `init.py`
   (header), `update_check.py` (user-agent + newer-than check). All import
   `from aai_cli import __version__`; the symbol must keep working.

Verified non-issues:

- **No syrupy snapshot pins the literal version** (`grep -r 0.1.4
  tests/__snapshots__/` is empty), so dynamic dev versions (`0.1.5.devN`) will not
  destabilize snapshot tests. The banner renders `__version__` but is not snapshotted.
- **No test hardcodes `"0.1.4"`** — version tests compare against `__version__`
  itself, so dynamic versioning keeps them green.
- A clean `main` checkout sitting on the latest tag computes exactly that tag
  (e.g. `0.1.4`, no dev suffix); dev suffixes appear only on commits past a tag.

## Design

### Version source — `pyproject.toml`

- `[build-system].requires = ["hatchling", "hatch-vcs"]`
- `[project]`: remove `version = "0.1.4"`; add `dynamic = ["version"]`
- Add:
  ```toml
  [tool.hatch.version]
  source = "vcs"
  raw-options = { local_scheme = "no-local-version" }
  ```
  `no-local-version` makes dev builds report clean PEP 440 versions
  (`0.1.5.dev3`) rather than `0.1.5.dev3+g1a2b3c`.
- **No `version-file`.** Generating `aai_cli/_version.py` would drop an untracked
  module into the tree that ruff / mypy / pyright / vulture / import-linter all
  have to special-case. Runtime reads installed metadata instead.

### Runtime `__version__` — `aai_cli/__init__.py`

Replace the literal with a metadata read:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aai-cli")
except PackageNotFoundError:  # not installed (bare source tree)
    __version__ = "0.0.0+unknown"
```

`importlib.metadata.version("aai-cli")` resolves for both the uv editable install
(always present under `uv run` / `uv sync`) and the Homebrew venv. All five
runtime consumers and every existing test continue to read `__version__` unchanged.

### Homebrew formula — `Formula/assembly.rb`

The GitHub tarball has no `.git`, so `def install` sets the setuptools-scm
pretend-version (hatch-vcs delegates to setuptools-scm) from the formula's parsed
`version` before installing:

```ruby
def install
  ENV["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AAI_CLI"] = version.to_s
  virtualenv_install_with_resources
end
```

With the pretend-version set, the built wheel records the tag version, so
`importlib.metadata` returns it and the `test do` assertion passes. The exact env
var name (`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AAI_CLI`, normalized dist name) is
verified during implementation.

`.github/workflows/release.yml` needs **no change** — it operates on tags + the
formula, not on a version string.

### Scripts

- **Delete `scripts/bump_patch.sh`** — there is no version string to bump.
- **Rewrite `scripts/cut_release.sh`** to take the version from the tag rather
  than reading files:
  - No arg → compute the next patch from the latest `vX.Y.Z` tag (preserves the
    current `bump_patch.sh` patch-bump ergonomics).
  - Explicit `cut_release.sh 0.2.0` → tag that version.
  - Keep all existing safety gates: must be on `main`, clean working tree, in sync
    with `origin/main`, tag must not already exist locally or on origin.
  - Keep `-n/--dry-run` and `-y/--yes`.

### Tests

- New test for the `PackageNotFoundError` fallback branch: monkeypatch
  `importlib.metadata.version` to raise `PackageNotFoundError` and assert
  `__version__` resolution yields `"0.0.0+unknown"`. Required for the 100%
  patch-coverage and mutation gates (the fallback line must be asserted, not just
  executed).
- Existing version tests (`test_smoke.py`, `test_telemetry.py`,
  `test_update_check.py`, `test_main_module.py`) stay green unchanged.

### Docs — `CLAUDE.md`

Update the "Naming & packaging gotchas" + release paragraphs: drop the dual-file
lock-step description, document tag-derived versioning, the deleted
`bump_patch.sh`, the new `cut_release.sh` behavior, and the formula's
pretend-version requirement.

## Build sequence

1. **Spike `uv lock --check` with the dynamic version.** Convert
   `pyproject.toml` to dynamic VCS version, run `uv lock` then `uv lock --check`
   on a dev commit (past the tag). Confirm the recorded project version does not
   drift the check red.
   - **Fallback if it drifts:** pin the project version for lock/dev via
     `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AAI_CLI` in the dev environment, or
     document a `uv lock` refresh step. This step **gates everything else** — if
     the fallback is unacceptable, revisit the approach before proceeding.
2. `pyproject.toml` build-system + dynamic version + `[tool.hatch.version]`.
3. `aai_cli/__init__.py` metadata read + fallback test.
4. `Formula/assembly.rb` pretend-version in `def install`.
5. Delete `bump_patch.sh`; rewrite `cut_release.sh`.
6. Update `CLAUDE.md`.
7. Run the full gate (`./scripts/check.sh`) to green.

## Out of scope

- Publishing to PyPI (the `assemblyai-cli` / `aai-cli` name is squatted; releases
  are Homebrew-only).
- Changing the release CI workflow structure (bottle build, GH release, formula PR).
- Conventional-commits / changelog automation (`python-semantic-release`,
  `release-please`) — a separate future decision.
