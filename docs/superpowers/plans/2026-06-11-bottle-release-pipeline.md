# Bottle Release Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a prebuilt arm64 Homebrew bottle on every `vX.Y.Z` tag so `brew install assembly` / `brew upgrade assembly` is fast (no Rust/LLVM source build), and remove the `install.sh` one-liner so the install story is just Homebrew (primary) + pipx/uv `git+https` (fallback).

**Architecture:** A tag-triggered `release.yml` with two jobs — `bottle` (macOS arm64: pin the formula to the tag's source, build the bottle, merge the `bottle do` block into the formula) and `publish` (Linux: create the GitHub Release with the bottle attached, open a formula PR a maintainer merges by hand). No special token: the built-in `GITHUB_TOKEN` opens the PR; CI doesn't auto-run on `GITHUB_TOKEN` PRs, so the maintainer merges with the admin override. Separately, delete `install.sh` and everything that exists only to support it.

**Tech Stack:** GitHub Actions, Homebrew (`brew bottle`/`brew tap-new`), Python `re` for formula edits, `gh` CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-bottle-release-pipeline-design.md`

---

## File Structure

**Created:**
- `.github/workflows/release.yml` — the tag-triggered release pipeline (two jobs).

**Modified:**
- `Formula/assembly.rb` — unchanged by hand; the pipeline rewrites `url`/`sha256` and inserts the `bottle do` block at release time. (No edit in this plan.)
- `pyproject.toml` — drop the `install_script` marker registration + its `addopts` exclusion.
- `tests/conftest.py` — drop `install_script` from `_NETWORK_MARKERS`.
- `.github/workflows/ci.yml` — delete the `install-smoke` job.
- `scripts/check.sh` — drop `install.sh` from the `shellcheck` arg list, drop `install_script` from the pytest `-m` filter, update comments.
- `scripts/mutation_gate.py` — drop `install_script` from `_DEFAULT_MARKERS`.
- `README.md` — delete the "One-liner" section; keep the pipx/uv section; note the bottle in the Homebrew section.
- `AGENTS.md` — drop `install_script` from the marker docs (two spots).
- `.claude/skills/check/SKILL.md` — drop the `install_script` line + `install.sh` mention.
- `.claude/skills/release-prep/SKILL.md` — rewrite for the tag→bottle flow.

**Deleted:**
- `install.sh`
- `tests/test_install_sh.py`
- `tests/test_install_script_smoke.py`

---

## Task 1: Remove `install.sh` and its support

This is a cohesive removal — the tree is only green once *all* references are gone, so the edits land in one commit verified by the full gate. Do the edits, then run `./scripts/check.sh`, then commit.

**Files:**
- Delete: `install.sh`, `tests/test_install_sh.py`, `tests/test_install_script_smoke.py`
- Modify: `pyproject.toml`, `tests/conftest.py`, `.github/workflows/ci.yml`, `scripts/check.sh`, `scripts/mutation_gate.py`, `README.md`, `AGENTS.md`, `.claude/skills/check/SKILL.md`

- [ ] **Step 1: Delete the script and its two test files**

```bash
git rm install.sh tests/test_install_sh.py tests/test_install_script_smoke.py
```

- [ ] **Step 2: Drop the `install_script` marker from `pyproject.toml`**

In `pyproject.toml`, change the `addopts` line (currently line ~131):

```toml
addopts = "--disable-socket --allow-unix-socket -m 'not e2e and not install and not install_script'"
```

to:

```toml
addopts = "--disable-socket --allow-unix-socket -m 'not e2e and not install'"
```

Then delete the `install_script` entry from the `markers = [` list (the line beginning `"install_script: real install of a locally-built wheel via install.sh …"`). Also delete/adjust the comment block just above `addopts` that names `install_script` (the lines starting `# install_script marker suites are slow…`) so it reads about `e2e`/`install` only — don't leave a comment referencing a marker that no longer exists (vulture/grep won't catch a stale comment, but the gate's markdown/style won't either; keep docs honest).

- [ ] **Step 3: Drop `install_script` from the conftest network markers**

In `tests/conftest.py` (line ~18):

```python
_NETWORK_MARKERS = ("e2e", "install", "install_script")
```

to:

```python
_NETWORK_MARKERS = ("e2e", "install")
```

- [ ] **Step 4: Delete the `install-smoke` CI job**

In `.github/workflows/ci.yml`, delete the entire `install-smoke:` job (starts at the `install-smoke:` key, line ~193, through the end of its last step — the line before `formula-install:` at line ~235). `formula-install:` and everything else stays.

- [ ] **Step 5: Update `scripts/check.sh`**

Three edits:

1. The `shellcheck` invocation (line ~125) — drop `install.sh`:

```sh
  shellcheck install.sh scripts/check.sh scripts/docker_build_check.sh
```

to:

```sh
  shellcheck scripts/check.sh scripts/docker_build_check.sh
```

2. The shellcheck echo just above (line ~121) `echo "==> shellcheck (install.sh)"` → `echo "==> shellcheck"`.

3. The pytest line (line ~181) — drop `and not install_script`:

```sh
uv run pytest -q --strict-config --strict-markers -n auto -m "not e2e and not install and not install_script" --cov=aai_cli --cov-branch --cov-context=test --cov-report=term-missing --cov-report=xml --cov-fail-under=90
```

to:

```sh
uv run pytest -q --strict-config --strict-markers -n auto -m "not e2e and not install" --cov=aai_cli --cov-branch --cov-context=test --cov-report=term-missing --cov-report=xml --cov-fail-under=90
```

4. Update the two comments naming `install_script` (lines ~172 and ~224) to drop it — e.g. the "and install_script (builds a wheel and runs install.sh for real…" comment, and the "env-gated marker suites (e2e/install/install_script)…" comment → "(e2e/install)".

- [ ] **Step 6: Update `scripts/mutation_gate.py`**

Line ~32:

```python
_DEFAULT_MARKERS = "not e2e and not install and not install_script"
```

to:

```python
_DEFAULT_MARKERS = "not e2e and not install"
```

- [ ] **Step 7: Remove the README "One-liner" section**

In `README.md`, delete the entire block (lines ~52–58):

```markdown
### One-liner

```sh
curl -fsSL https://raw.githubusercontent.com/AssemblyAI/cli/main/install.sh | sh
```

Prefers [`pipx`](https://pipx.pypa.io), falling back to `pip --user`.
```

Leave the `### pipx / uv` section above it intact (it's the fallback path).

- [ ] **Step 8: Update `AGENTS.md`**

`AGENTS.md` is the source; `CLAUDE.md` is a symlink to it, so edit `AGENTS.md` only.

1. The marker-sets sentence (line ~57): change "excludes **three** slow/credentialed marker sets — … `-m "not e2e and not install and not install_script"`" to "excludes **two** … `-m "not e2e and not install"`".
2. Delete the `uv run pytest -m install_script  # builds a wheel and runs install.sh for real; …` line (line ~62) from the opt-in examples block.
3. The hermetic-suite paragraph (line ~71): change the two `e2e`/`install`/`install_script` enumerations to `e2e`/`install`.

- [ ] **Step 9: Update `.claude/skills/check/SKILL.md`**

1. In the gate-order sentence (line ~19): change `shellcheck install.sh scripts/check.sh` → `shellcheck scripts/check.sh`, and change "excluding `e2e`/`install`/`install_script` markers" → "excluding `e2e`/`install` markers".
2. Delete the `uv run pytest -m install_script  # builds a wheel and runs install.sh for real; needs network + uv/pipx` line (line ~31).

- [ ] **Step 10: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` — deptry/vulture/ruff confirm no orphaned imports or dead code; the snapshot tests and `--strict-markers` confirm nothing still references the removed marker.

If a `--help` or other syrupy snapshot shifts (it shouldn't — no command output changed), do **not** hand-edit it; regenerate with `uv run pytest --snapshot-update` and re-run the gate.

- [ ] **Step 11: Confirm no stray references remain**

Run:

```bash
grep -rn "install\.sh\|install_script" --include="*.py" --include="*.yml" --include="*.toml" --include="*.md" --include="*.sh" . | grep -v "/.venv/" | grep -v "docs/superpowers/specs/2026-06-04" | grep -v "docs/superpowers/specs/2026-06-08" | grep -v "docs/superpowers/plans/2026-06-04" | grep -v "docs/superpowers/plans/2026-06-08" | grep -v "docs/superpowers/specs/2026-06-11-bottle-release-pipeline" | grep -v "docs/superpowers/plans/2026-06-11-bottle-release-pipeline" | grep -v "release-prep"
```

Expected: **no output** (the only remaining mentions are the dated historical specs/plans, this plan/spec, and `release-prep` which Task 3 rewrites).

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "Remove install.sh one-liner and its support

Homebrew (bottle) + pipx/uv git+https are the supported install paths;
drop the curl|sh installer, its unit + smoke tests, the install_script
marker, and the install-smoke CI job.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add the `release.yml` bottle pipeline

A workflow can only be fully exercised by a tag push, so verification here is: it parses, passes `actionlint` + `zizmor` (the same gates `check.sh`/CI apply to workflows), and `Formula/assembly.rb` still passes `brew style` after a local `--merge` dry check. The first real `v0.1.0` tag is the integration test (see Task 4).

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Resolve SHA pins for the artifact actions**

`release.yml` adds `actions/upload-artifact` and `actions/download-artifact`, which `ci.yml` doesn't use yet. `zizmor` requires SHA-pinned `uses:`, so resolve the v4 tag SHAs at implementation time (don't guess):

```bash
gh api repos/actions/upload-artifact/git/ref/tags/v4 --jq '.object.sha'
gh api repos/actions/download-artifact/git/ref/tags/v4 --jq '.object.sha'
```

Note the two SHAs; substitute them for `<UPLOAD_ARTIFACT_SHA>` / `<DOWNLOAD_ARTIFACT_SHA>` in Step 2. (If `git/ref/tags/v4` 404s because v4 is a moving major tag pointing at a SHA via an annotated tag, use `gh api repos/actions/upload-artifact/git/refs/tags/v4 --jq '.[0].object.sha'` or read the SHA from the action's latest v4.x release.)

- [ ] **Step 2: Write `.github/workflows/release.yml`**

Create `.github/workflows/release.yml` with this content (fill in the four SHAs — reuse `ci.yml`'s pins for `checkout`/`setup-homebrew`, and the two resolved in Step 1):

```yaml
name: Release

# Cut a release by pushing a vX.Y.Z tag (after the version-bump PR merges).
# Builds the arm64 macOS bottle, publishes it to the tag's GitHub Release, and
# opens a formula PR (url + sha256 + bottle block) for a maintainer to merge.
on:
  push:
    tags: ["v*"]
  # Manual dry-run: build the bottle for an existing tag WITHOUT publishing or
  # opening a PR (publish steps are gated on a real tag push below).
  workflow_dispatch:
    inputs:
      tag:
        description: "Existing tag to build a bottle for (dry-run; no publish)"
        required: true

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.event.inputs.tag || github.ref }}
  cancel-in-progress: false

jobs:
  bottle:
    name: build arm64 bottle (macOS)
    # Pin to the OLDEST supported arm64 macOS on purpose: Homebrew uses the
    # newest bottle whose tag is <= the running OS, so an arm64_sonoma bottle is
    # auto-selected on Sonoma AND every newer macOS. Building on newer would not
    # serve older. macos-14 == Sonoma/arm64.
    runs-on: macos-14
    timeout-minutes: 40
    permissions:
      contents: read
    outputs:
      tag: ${{ steps.meta.outputs.tag }}
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6.0.3
        with:
          persist-credentials: false # this job doesn't push
      - uses: Homebrew/actions/setup-homebrew@2ebcf16054461267868620b1414507f3ccc765c1

      - name: Resolve tag + source sha256
        id: meta
        run: |
          set -euo pipefail
          tag="${{ github.event.inputs.tag || github.ref_name }}"
          url="https://github.com/${GITHUB_REPOSITORY}/archive/refs/tags/${tag}.tar.gz"
          curl -fL "$url" -o source.tar.gz
          sha="$(shasum -a 256 source.tar.gz | awk '{print $1}')"
          {
            echo "tag=${tag}"
            echo "source_sha=${sha}"
            echo "root_url=https://github.com/${GITHUB_REPOSITORY}/releases/download/${tag}"
          } >> "$GITHUB_OUTPUT"

      - name: Pin the formula to the release tag
        env:
          TAG: ${{ steps.meta.outputs.tag }}
          SOURCE_SHA: ${{ steps.meta.outputs.source_sha }}
          REPO: ${{ github.repository }}
        run: |
          set -euo pipefail
          python3 - <<'PY'
          import os, re, pathlib
          tag, sha, repo = os.environ["TAG"], os.environ["SOURCE_SHA"], os.environ["REPO"]
          url = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
          p = pathlib.Path("Formula/assembly.rb")
          src = p.read_text()
          # count=1: only the stable url/sha256 (the first sha256 in the file);
          # every `resource` block's sha256 stays untouched.
          src = re.sub(r'url ".*?"', f'url "{url}"', src, count=1)
          src = re.sub(r'sha256 "[0-9a-f]*"', f'sha256 "{sha}"', src, count=1)
          p.write_text(src)
          PY
          grep -nE '^  (url|sha256) ' Formula/assembly.rb | head -2

      - name: Build the bottle + merge the block into the formula
        env:
          ROOT_URL: ${{ steps.meta.outputs.root_url }}
        run: |
          set -euo pipefail
          # Newer Homebrew refuses formulae outside a tap; use a throwaway local one.
          brew tap-new --no-git assembly/local
          tap_formula="$(brew --repository assembly/local)/Formula/assembly.rb"
          cp Formula/assembly.rb "$tap_formula"
          brew install --build-bottle --formula assembly/local/assembly
          brew bottle --json --no-rebuild --root-url="$ROOT_URL" assembly/local/assembly
          # brew writes a double-dash local name; the canonical download name is
          # single-dash. Rename the tarball; the JSON already records both names.
          for f in assembly--*.bottle.tar.gz; do mv "$f" "${f/--/-}"; done
          # Merge the bottle block into the TAP copy (where the formula lives),
          # then copy the finalized formula back to the repo path for upload.
          brew bottle --merge --write --no-commit assembly--*.bottle.json
          cp "$tap_formula" Formula/assembly.rb
          echo "--- finalized formula head ---"
          sed -n '1,20p' Formula/assembly.rb

      - name: Upload bottle + finalized formula
        uses: actions/upload-artifact@<UPLOAD_ARTIFACT_SHA> # v4
        with:
          name: release-artifacts
          path: |
            assembly-*.bottle.tar.gz
            Formula/assembly.rb
          if-no-files-found: error

  publish:
    name: publish release + open formula PR
    needs: [bottle]
    # Real publish only on a tag push; workflow_dispatch is a build-only dry-run.
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    timeout-minutes: 10
    permissions:
      contents: write # create the release + push the formula branch
      pull-requests: write # open the formula PR
    steps:
      # NOTE: default persist-credentials (true) on purpose — this job pushes a
      # branch, so it needs the token wired into git.
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6.0.3

      - uses: actions/download-artifact@<DOWNLOAD_ARTIFACT_SHA> # v4
        with:
          name: release-artifacts
          path: artifacts

      - name: Create the GitHub Release with the bottle attached
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ needs.bottle.outputs.tag }}
        run: |
          set -euo pipefail
          # Bottle must live at <root_url>/<filename> == the release download path.
          gh release create "$TAG" \
            --title "$TAG" \
            --generate-notes \
            artifacts/assembly-*.bottle.tar.gz

      - name: Open the formula PR
        env:
          GH_TOKEN: ${{ github.token }}
          TAG: ${{ needs.bottle.outputs.tag }}
        run: |
          set -euo pipefail
          branch="release/${TAG}-formula"
          cp artifacts/Formula/assembly.rb Formula/assembly.rb
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git checkout -b "$branch"
          git add Formula/assembly.rb
          git commit -m "Bottle ${TAG}: pin url + sha256, add arm64_sonoma bottle"
          git push origin "$branch"
          gh pr create --base main --head "$branch" \
            --title "Bottle ${TAG}" \
            --body "Automated by release.yml: pins the formula to the ${TAG} source tarball and adds the arm64_sonoma \`bottle do\` block.

          **Merge with the admin override** (\"merge without waiting for requirements\"): a PR opened by \`GITHUB_TOKEN\` does not trigger CI, so the required \`lint + typecheck + tests\` check will not report on its own. The diff is formula-only."
```

- [ ] **Step 3: Lint the workflow**

If `actionlint` + `zizmor` are installed locally (the Go binaries `check.sh` self-skips when absent):

Run: `actionlint .github/workflows/release.yml && zizmor .github/workflows/release.yml`
Expected: no findings.

If they're not installed locally, run the full gate (`./scripts/check.sh`) — it runs both over `.github/workflows/` — or rely on CI's workflow-lint on the PR. Expected: clean.

- [ ] **Step 4: Sanity-check the formula still styles after a local merge (optional, macOS only)**

On a macOS box with Homebrew, dry-run the pin+merge against the existing tag-less formula to confirm `brew style` accepts the generated block. (Skippable on Linux/CI; the first real release validates end-to-end.)

```bash
cp Formula/assembly.rb /tmp/assembly.bak.rb
# simulate: leave url/sha placeholders, just check style baseline
brew style ./Formula/assembly.rb
```
Expected: PASS (baseline). Restore if you edited: `cp /tmp/assembly.bak.rb Formula/assembly.rb`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "Add tag-triggered bottle release pipeline

release.yml builds an arm64 macOS bottle on a vX.Y.Z tag, publishes it to
the GitHub Release, and opens a formula PR (url+sha256+bottle block) for a
maintainer to merge. Built-in GITHUB_TOKEN only; no special secret.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Rewrite the `release-prep` skill + note the bottle in the README

Bring the docs in line with the new flow: a release is a version-bump PR, a tag push, and a formula-PR merge — the bottle/release artifacts are produced by `release.yml`, not by hand.

**Files:**
- Modify: `.claude/skills/release-prep/SKILL.md`, `README.md`

- [ ] **Step 1: Rewrite `.claude/skills/release-prep/SKILL.md`**

Replace the file body with the tag→bottle flow. Key changes: drop the `uv run pytest -m install_script` step (Section 3) entirely; replace the "Distribution caveat" mention of `install.sh` with Homebrew-bottle + pipx/uv. New content:

```markdown
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
git tag vX.Y.Z
git push origin vX.Y.Z
```

The push triggers `.github/workflows/release.yml`, which:
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
```

- [ ] **Step 2: Note the bottle in the README Homebrew section**

In `README.md`, the Homebrew prose currently reads:

```markdown
`brew install` pulls in `ffmpeg` and `portaudio` for you, so `transcribe`, `stream`, and `agent` work out of the box. Upgrade with `brew upgrade assembly`; remove with `brew uninstall assembly`.
```

Append a sentence so users know upgrades are prebuilt:

```markdown
`brew install` pulls in `ffmpeg` and `portaudio` for you, so `transcribe`, `stream`, and `agent` work out of the box. Releases ship a prebuilt arm64 bottle, so `brew install`/`brew upgrade assembly` is a fast binary install (no compiler toolchain); Intel Macs build from source or can use the pipx/uv path below. Remove with `brew uninstall assembly`.
```

- [ ] **Step 3: Run markdownlint + the gate**

Run: `./scripts/check.sh`
Expected: `All checks passed.` (markdownlint covers the README + SKILL.md edits; the snapshot suite is unaffected — no command output changed).

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/release-prep/SKILL.md README.md
git commit -m "Docs: release-prep + README reflect the bottle release flow

Release is now: version-bump PR -> tag push -> release.yml builds the
bottle + opens the formula PR -> admin-merge. Drop the install.sh smoke
step and the squatted-PyPI install.sh caveat.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: First-release integration verification (manual, post-merge)

Not a code change — the end-to-end proof that the pipeline works. Run after Tasks 1–3 are merged to `main` and you're ready to cut the first real version.

- [ ] **Step 1: Bump + tag**

Follow the rewritten `release-prep` skill: bump `pyproject.toml` to `0.1.0` via PR, merge, then `git tag v0.1.0 && git push origin v0.1.0`.

- [ ] **Step 2: Watch the workflow**

Run: `gh run watch` (or the Actions tab). Expected: `bottle` job green (bottle built + merged), `publish` job green (release created, `release/v0.1.0-formula` PR opened).

- [ ] **Step 3: Merge the formula PR (admin override)**

Run: `gh pr view release/v0.1.0-formula --web` → review the formula-only diff → merge with "merge without waiting for requirements".

- [ ] **Step 4: Confirm the bottle is used**

On a clean arm64 Mac:

Run:
```bash
brew untap assemblyai/cli 2>/dev/null || true
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew trust assemblyai/cli
brew install assembly
assembly --version
```
Expected: install is a fast bottle pour with **no** `rust`/`llvm`/`python@3.14` pulled, and `assembly --version` prints `0.1.0`. Confirm with `brew deps assembly` — `rust`/`pkgconf` should be absent (they were `:build`-only and a bottle skips the build).

---

## Self-Review

**Spec coverage:**
- arm64 bottle on tag → Task 2 (`bottle` job). ✅
- GitHub Releases hosting / `root_url` → Task 2 (`publish` job, `gh release create`). ✅
- No special token / maintainer-merged PR → Task 2 (`publish` PR step), Task 3 (release-prep step 4). ✅
- Ordering problem (url/sha256/bottle-block post-tag) → Task 2 (pin + merge steps). ✅
- Remove install.sh + support → Task 1. ✅
- pipx/uv stay (`git+https`) → Task 1 keeps the README section; Task 3 caveat. ✅
- No wheels / no moving tag / no PEP 503 index → not built (correctly absent). ✅
- Conventions (SHA pins, persist-credentials, timeouts, concurrency) → Task 2 Steps 1–2. ✅
- Testing strategy (actionlint/zizmor, gate green, first-release proof) → Task 2 Step 3, Task 1 Step 10, Task 4. ✅
- Release runbook → Task 3 (release-prep rewrite). ✅

**Placeholder scan:** The only intentional placeholders are `<UPLOAD_ARTIFACT_SHA>` / `<DOWNLOAD_ARTIFACT_SHA>`, resolved in Task 2 Step 1 before use. No TBD/TODO.

**Consistency:** Artifact name `release-artifacts` is uploaded (bottle job) and downloaded (publish job) with the same name; `needs.bottle.outputs.tag` is declared as a `bottle` job output and consumed in `publish`. Bottle filename `assembly-*.bottle.tar.gz` (single-dash, post-rename) is the form uploaded, released, and matched by `root_url`. Branch name `release/${TAG}-formula` is consistent across Task 2 and Tasks 3–4.
