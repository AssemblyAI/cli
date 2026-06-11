# Bottle release pipeline — tagged releases that ship a prebuilt Homebrew bottle

**Date:** 2026-06-11
**Status:** Approved design

## Summary

Add a tag-triggered GitHub Actions workflow (`release.yml`) that, on a pushed
`vX.Y.Z` tag, builds the release artifacts and finalizes the Homebrew formula so
`brew install assembly` downloads a **prebuilt bottle** instead of compiling the
Rust-backed dependencies from source.

The whole motivation: the formula vendors source distributions (sdists), so a
from-source `brew install` drags in `rust` → `llvm` → `z3` / `libgit2` / a second
`python@3.14` purely to compile `pydantic-core`, `jiter`, and `cryptography`. A
bottle is a tarball of the already-built Cellar tree — installing it needs none of
that toolchain. The pip/pipx/uv paths are unaffected (they already pull prebuilt
dependency wheels from PyPI and never invoke Rust), so this is a Homebrew-only fix.
The same workflow also builds the Python wheel + sdist and attaches them to the
GitHub Release, giving the pip/pipx paths version-pinned install targets.

### Scope decisions (locked during brainstorming)

- **Full release pipeline**, not just a bottle step — one tag push produces the
  GitHub Release, the Python artifacts, the bottle, and the finalized formula.
- **arm64 macOS bottle only.** macOS is where `brew install` is the recommended
  path; Apple Silicon is the dominant Mac base. Intel Mac and all of Linux fall
  back to the existing from-source build (still works), and Linux users are
  steered to pipx/uv by the README anyway. Easy to add `x86_64_linux` later.
- **Bottles hosted on GitHub Releases** (a `root_url` on the release download
  path), not ghcr.io — simplest for a single self-tap on a public repo, no extra
  auth.
- **No special token.** The workflow uses only the built-in `GITHUB_TOKEN`; the
  finalized-formula change lands via a PR a maintainer merges by hand (see
  "Committing the formula back to main").

## Background: the ordering problem

A bottled formula needs three values that do not all exist until *after* the tag
is pushed:

1. `url` — the `vX.Y.Z` source-tarball URL
   (`https://github.com/AssemblyAI/cli/archive/refs/tags/vX.Y.Z.tar.gz`).
2. `sha256` — that source tarball's checksum. GitHub generates the archive on
   demand; its bytes are only pinnable once the tag exists.
3. `bottle do … end` — the bottle's own checksum, only known after the bottle is
   built (which itself requires the formula to already point at a real, downloadable
   `url` + `sha256`).

Today the committed formula carries placeholders for (1)/(2) — `url` points at a
not-yet-cut `v0.1.0` and `sha256` is `"0" * 64` — and has no bottle block. The
workflow computes all three after the tag push and writes them back in one commit.

## Trigger and human workflow

`release.yml` triggers on `push:` with `tags: ["v*"]`.

A release is two manual steps; everything else is automated:

1. Bump the version in `pyproject.toml` to `X.Y.Z` via a normal PR (regular CI).
2. Push the tag `vX.Y.Z` on `main`.

Then one manual click at the end: merge the formula PR the workflow opens
(see below).

## Architecture — three jobs

### Job 1 · `pypi-artifacts` (ubuntu-latest)

`python -m build` → `dist/*.whl` + `dist/*.tar.gz`. Upload as a workflow artifact.
This is the pip/pipx version-pinning payoff (installable, pinned release assets
rather than only `git+https://…@main`).

### Job 2 · `bottle` (macos-14)

Pinned to the **oldest supported arm64 macOS on purpose**: Homebrew uses the
newest bottle whose macOS tag is ≤ the running OS, so a bottle built on Sonoma
(`arm64_sonoma`) is auto-selected on Sonoma *and* every newer macOS (Sequoia,
Tahoe, …). One build covers the whole range; building on the newest OS would
*not* serve older ones.

Steps:

1. Check out the repo at the tag (`persist-credentials: false`, like every other
   job).
2. **Compute the source sha256:** `curl -fL` the tag archive
   (`…/archive/refs/tags/vX.Y.Z.tar.gz`) and `shasum -a 256`. Curling GitHub's
   actual generated archive (rather than reconstructing it with `git archive`)
   guarantees the checksum matches what end users' `brew install` downloads.
3. **Repoint the formula** `url` → the tag archive and `sha256` → the computed
   value (replacing the placeholders), leaving every `resource` block untouched
   (`count=1` style edits, mirroring the existing `formula-install` job's patch).
4. **Build the bottle** through a throwaway local tap (newer Homebrew refuses
   formulae outside a tap):
   - `brew tap-new --no-git assembly/local`
   - copy the patched `Formula/assembly.rb` into the local tap
   - `brew install --build-bottle --formula assembly/local/assembly`
   - `brew bottle --json --no-rebuild --root-url="https://github.com/AssemblyAI/cli/releases/download/vX.Y.Z" assembly/local/assembly`
5. **Rename** the produced `assembly--X.Y.Z.arm64_sonoma.bottle.tar.gz` (Homebrew
   writes a double dash locally) to the canonical single-dash download name
   `assembly-X.Y.Z.arm64_sonoma.bottle.tar.gz`.
6. Upload the renamed tarball **and** the `*.bottle.json` as artifacts.

### Job 3 · `publish` (ubuntu-latest, `needs: [pypi-artifacts, bottle]`)

1. Download both jobs' artifacts.
2. **Merge the bottle block into the formula:** `brew bottle --merge --write
   --no-commit <bottle>.json`, run against the same patched formula (url + real
   sha256). The formula now carries `url`, the real `sha256`, and a `bottle do`
   block with `root_url` + the `arm64_sonoma` checksum.
3. **Create the release:** `gh release create vX.Y.Z` (or `gh release upload` if
   it already exists) attaching the wheel, sdist, and the renamed bottle tarball.
   The bottle must live at `<root_url>/<bottle-filename>`, which is exactly the
   release download path, so the `root_url` and the upload target agree.
4. **Open the formula PR** (see below).

## Committing the formula back to `main`

`main` is branch-protected with a required `lint + typecheck + tests` check (the
`ci.yml` comments confirm this) and almost certainly "require a PR before
merging". The finalized formula (url + sha256 + bottle block) must live on `main`
because that is the branch users tap.

A direct push from CI is blocked by protection. Auto-merging a PR is blocked by a
GitHub anti-recursion rule: **a PR opened by `GITHUB_TOKEN` does not trigger other
workflows**, so `ci.yml`'s `pull_request` jobs never run on it, the required check
never reports, and auto-merge would hang forever.

**Chosen approach (no special token):** Job 3 opens a PR from a
`release/vX.Y.Z-formula` branch using the built-in `GITHUB_TOKEN`
(`permissions: contents: write, pull-requests: write`). Because that PR's checks
don't auto-run, a **maintainer merges it manually** using the repo-admin "merge
without waiting for requirements" override. The formula-only diff is small and
glanceable, so a human approve-and-merge is appropriate. Zero stored secrets; one
extra click per release.

Explicitly rejected alternative: a fine-grained PAT or GitHub App in the
branch-protection bypass list. It enables full hands-off auto-merge but introduces
a privileged stored secret and punches a hole in branch protection — not worth it
for a release cadence that already has two manual steps.

## Conventions to follow (match existing `ci.yml`)

- Pin every action to a commit SHA with a `# vX.Y.Z` comment; Dependabot keeps
  them current. Reuse the SHAs already pinned in `ci.yml`
  (`actions/checkout`, `Homebrew/actions/setup-homebrew`).
- `persist-credentials: false` on every checkout that doesn't push; scope
  `permissions:` to the minimum per job (`contents: read` default; `contents:
  write` + `pull-requests: write` only on `publish`).
- `timeout-minutes` on every job (the from-source bottle build compiles Rust, so
  `bottle` needs a generous timeout — model it on `formula-install`'s 40).
- `concurrency` group keyed on the tag so a re-pushed tag supersedes cleanly.

## Testing strategy

The pipeline can only be fully exercised by pushing a real tag, so validation is
layered:

- **`brew style` already gates the formula** on every PR (`lint-formula` job). The
  finalized formula with a `bottle do` block must keep passing `brew style`; if the
  generated block needs reformatting, the merge step normalizes it.
- **`formula-install` (existing) stays the from-source correctness check.** It
  already builds the branch's formula from source on macOS and runs `brew test`.
  The bottle is the *same* build packaged, so this job continues to prove the
  resource list installs and `assembly --version` works.
- **Workflow lint:** `release.yml` must pass `actionlint` + `zizmor`, which
  `check.sh` already runs over `.github/workflows/`.
- **First real release is the integration test.** Cutting `v0.1.0` exercises the
  end-to-end path; a follow-up `brew install assembly` on a clean arm64 Mac
  confirms the bottle is selected (no `rust`/`llvm` pulled). This is a manual
  post-merge verification step, documented in the release runbook.
- **Optional dry-run hook:** support `workflow_dispatch` on `release.yml` so the
  build-bottle path can be run against an existing tag without re-tagging, for
  debugging, gated so it never publishes/commits.

## Release runbook (to document in repo)

1. `pyproject.toml` version bump → PR → merge.
2. `git tag vX.Y.Z && git push origin vX.Y.Z`.
3. Wait for `release.yml`; review and merge the `release/vX.Y.Z-formula` PR
   (admin override merge).
4. Verify on a clean arm64 Mac: `brew update && brew install assembly` pulls the
   bottle (fast, no `rust`); `assembly --version` matches.

## Out of scope

- Linux (`x86_64_linux`) and Intel-Mac (`ventura`) bottles — additive later.
- Publishing to PyPI (the `assemblyai-cli` name is squatted; release assets live
  on GitHub Releases instead).
- ghcr.io / OCI bottle hosting.
- Automating the version bump itself (stays a human PR).
