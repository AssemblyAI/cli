# Homebrew Tap (virtualenv formula, in-repo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute the `aai` CLI via `brew install` using a Homebrew `Language::Python::Virtualenv` formula that lives inside this repo (the repo *is* the tap), sourcing the package from a GitHub release tag and never touching PyPI for our own package.

**Architecture:** A `Formula/aai.rb` file is committed to `AssemblyAI/cli`. Users run `brew tap assemblyai/cli https://github.com/AssemblyAI/cli` then `brew install aai`. Homebrew builds an isolated virtualenv from our package's source tarball (the `v0.1.0` git-tag tarball) and installs all runtime dependencies from pinned `resource` stanzas (sourced from PyPI — those are third-party packages, we publish nothing). Native dependencies (`portaudio`, `openssl`) are declared via `depends_on` and provided by Homebrew rather than bundled.

**Tech Stack:** Homebrew (Ruby formula DSL), `Language::Python::Virtualenv`, Python 3.13, `brew update-python-resources`, hatchling (our build backend), GitHub Releases.

---

## Decisions & Risks (read before starting)

- **Source artifact = GitHub tag tarball.** `url` → `https://github.com/AssemblyAI/cli/archive/refs/tags/v0.1.0.tar.gz` (the tarball GitHub auto-generates per tag). Idiomatic (httpie/azure-cli), zero per-release asset upload. Not PyPI.
- **Release is on the critical path (bootstrap only).** The main `sha256` requires the tag to exist, and `brew update-python-resources` requires a resolvable `url`. So Task 2 (cut `v0.1.0` *by hand*) gates Tasks 3–4. This manual step happens **once**; from then on Task 6's pipeline cuts every release automatically on merge to `main` (Conventional Commits → SemVer via `python-semantic-release`, then an auto formula bump). No PyPI publish in either the bootstrap or the steady state.
- **Resources come from `brew update-python-resources`, not a lockfile dump.** It correctly handles platform-conditional deps. The 48-package runtime closure includes Linux-only (`jeepney`, `secretstorage`) and Windows-only (`pywin32-ctypes`) packages that must NOT be unconditional `resource` blocks. Appendix A has an offline `uv.lock` generator as a fallback, with this caveat called out.
- **Native build dependencies.** Closure contains Rust-built (`pydantic-core`, `jiter`, `cryptography`) and C-extension (`cffi`) packages, plus `sounddevice` which needs PortAudio. Formula declares `rust`/`pkgconf` (build), `openssl@3`, `portaudio`, `python@3.13`. The `cryptography` source build is slow; Appendix B documents the `depends_on "cryptography" => :no_linkage` optimization if the build is too painful.
- **RISK — `aai` binary name collision.** Your `alexkroman/aai` tap installs a command also named `aai`. Both cannot coexist in one Homebrew prefix. This plan keeps `aai` (matches `[project.scripts] aai = "assemblyai_cli.main:run"`). Resolving the collision (rename command, or rename one tap's formula) is out of scope and tracked as an open product decision.
- **RISK — `sounddevice` runtime linkage.** Built from sdist, `sounddevice` binds PortAudio via cffi/`ctypes.util.find_library`. With `depends_on "portaudio"` it should resolve, but this is the most likely runtime failure; Task 4 explicitly tests the audio import path, not just `--version`.
- **Branch hygiene.** You are on `stytch-oauth-cli-login` with an uncommitted `auth/endpoints.py`. The release tag (Task 2) must come off `main`. Do all formula/script work on a normal feature branch; do not tag from the feature branch.

---

## File Structure

- `Formula/aai.rb` — the Homebrew formula (created). Class `Aai`. Homebrew auto-discovers `Formula/` in a tapped repo. No explicit `version` line — Homebrew parses it from the tag in `url`, so the auto-bump only rewrites `url` + `sha256`.
- `scripts/generate_brew_resources.py` — offline fallback generator from `uv.lock` (created; Appendix A). Not the primary path.
- `README.md` — add a Homebrew install section (modified).
- `pyproject.toml` — add `[tool.semantic_release]` config (modified; Task 6).
- `.github/workflows/release.yml` — release-on-merge pipeline: semantic-release cuts the version, then a guarded job bumps the formula (created; Task 6).

---

## Task 1: Scaffold the in-repo tap and a buildable formula skeleton (no resources yet)

**Files:**
- Create: `Formula/aai.rb`

- [ ] **Step 1: Create the formula skeleton**

Create `Formula/aai.rb` exactly:

```ruby
class Aai < Formula
  include Language::Python::Virtualenv

  desc "Command-line interface for AssemblyAI"
  homepage "https://github.com/AssemblyAI/cli"
  url "https://github.com/AssemblyAI/cli/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0" * 64 # FILLED IN TASK 2 once the tag exists
  license "MIT"

  depends_on "pkgconf" => :build      # cffi / cryptography native builds
  depends_on "rust" => :build         # pydantic-core, jiter, cryptography
  depends_on "openssl@3"              # cryptography linkage
  depends_on "portaudio"             # sounddevice (audio capture)
  depends_on "python@3.13"

  # RESOURCE STANZAS INSERTED IN TASK 3 (brew update-python-resources)

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/aai version")
  end
end
```

- [ ] **Step 2: Lint the skeleton for style**

Run: `brew style ./Formula/aai.rb`
Expected: PASS (0 offenses). If `brew style` reports offenses, fix exactly as it prints (it auto-describes each).

- [ ] **Step 3: Confirm the placeholder is the only audit blocker**

Run: `brew audit --new --formula ./Formula/aai.rb 2>&1 | head -20`
Expected: complaints about the dummy `sha256` and/or "no resources" — these are expected at this stage. There should be NO Ruby syntax errors. If you see "syntax error" or "uninitialized constant", the formula body is malformed — fix before continuing.

- [ ] **Step 4: Commit**

```bash
git add Formula/aai.rb
git commit -m "build(brew): add virtualenv formula skeleton for aai tap"
```

---

## Task 2: Cut the v0.1.0 release and fill the main sha256

**Files:**
- Modify: `Formula/aai.rb` (the `sha256` line)

> **Outward-facing — performed by the maintainer. One-time bootstrap.** This is the only hand-cut release; Task 6 automates all subsequent ones. Tag from `main`, not the feature branch. The `pyproject.toml` version is already `0.1.0`, so the tag matches; `python-semantic-release` will compute the *next* version by diffing commits against this `v0.1.0` tag.

- [ ] **Step 1: Tag and create the release off main**

```bash
git checkout main
git pull --ff-only
git tag v0.1.0
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes "Initial Homebrew release"
```

Expected: `gh release create` prints the release URL. Return to your working branch afterward: `git checkout -`.

- [ ] **Step 2: Compute the tarball sha256**

```bash
curl -fsSL "https://github.com/AssemblyAI/cli/archive/refs/tags/v0.1.0.tar.gz" | shasum -a 256
```

Expected: a 64-hex-char digest followed by `-`. Copy the digest.

- [ ] **Step 3: Insert the digest into the formula**

In `Formula/aai.rb`, replace the line `sha256 "0" * 64 # FILLED IN TASK 2 once the tag exists` with:

```ruby
  sha256 "<digest-from-step-2>"
```

- [ ] **Step 4: Verify Homebrew accepts the download**

Run: `brew fetch --formula ./Formula/aai.rb 2>&1 | tail -5`
Expected: "Downloaded to ..." with no checksum mismatch. A "SHA256 mismatch" error means the wrong digest was pasted — recompute.

- [ ] **Step 5: Commit**

```bash
git add Formula/aai.rb
git commit -m "build(brew): pin v0.1.0 source tarball checksum"
```

---

## Task 3: Generate the dependency resource stanzas

**Files:**
- Modify: `Formula/aai.rb` (insert `resource` blocks)

- [ ] **Step 1: Generate resources with the official tool**

Run: `brew update-python-resources --print-only ./Formula/aai.rb > /tmp/aai-resources.rb`
Expected: stdout-captured `resource "..." do ... end` blocks for the runtime closure (≈40+ blocks). The tool downloads our tag tarball, resolves deps, and emits PyPI sdist URLs + sha256.
If it errors with "could not find ... on PyPI" for *our* package `aai-cli`, that is fine to ignore — it resolves the dependencies, not our root package (which comes from `url`). If it cannot resolve at all, fall back to Appendix A.

- [ ] **Step 2: Insert the blocks into the formula**

Replace the line `  # RESOURCE STANZAS INSERTED IN TASK 3 (brew update-python-resources)` in `Formula/aai.rb` with the contents of `/tmp/aai-resources.rb` (indented two spaces to sit inside the class body).

- [ ] **Step 3: Verify platform-specific deps are scoped, not unconditional**

Run: `grep -nE "jeepney|secretstorage|pywin32-ctypes" Formula/aai.rb`
Expected: `pywin32-ctypes` should be ABSENT (Windows-only; Homebrew has no Windows). `jeepney`/`secretstorage` should either be absent or wrapped in `on_linux do ... end`. If `brew update-python-resources` emitted them unconditionally, move them inside an `on_linux do` block and delete `pywin32-ctypes`. (This is exactly why we do not use a raw lockfile dump.)

- [ ] **Step 4: Style + structural audit**

Run: `brew style ./Formula/aai.rb && brew audit --formula ./Formula/aai.rb 2>&1 | grep -vE "GitHub|stable|bottle" | head -20`
Expected: `brew style` PASS; audit shows no "duplicate resource" or "resource ... should be ..." naming errors. Fix any reported resource-name casing mismatches by matching the name the audit suggests.

- [ ] **Step 5: Commit**

```bash
git add Formula/aai.rb
git commit -m "build(brew): add pinned dependency resource stanzas"
```

---

## Task 4: Build, install, and functionally validate locally

**Files:**
- Modify: `Formula/aai.rb` (only if build reveals missing deps)

- [ ] **Step 1: Build from source**

Run: `brew install --build-from-source --verbose ./Formula/aai.rb 2>&1 | tail -30`
Expected: ends with "🍺 .../aai/0.1.0: ... files".
If a resource fails to build:
- `error: can't find Rust compiler` → ensure `depends_on "rust" => :build` present.
- `openssl`/`cryptography` build failure → confirm `depends_on "openssl@3"`; if still failing, apply Appendix B (`cryptography => :no_linkage`).
- `pkg-config`/`ffi` errors building `cffi` → confirm `depends_on "pkgconf" => :build`.
Add the missing `depends_on`, then re-run this step.

- [ ] **Step 2: Smoke-test the entrypoint**

Run: `aai version`
Expected: prints the version string (matches `0.1.0`). If `aai: command not found`, run `brew link aai` or check `$(brew --prefix)/bin` is on PATH.

- [ ] **Step 3: Test the audio path (the real risk)**

Run: `aai doctor 2>&1 | tail -20`
Expected: doctor output runs without an `ImportError`/`OSError: PortAudio library not found`. The critical check is that `import sounddevice` succeeds inside the bottled venv.
If PortAudio is not found at runtime: set the formula to expose it — add to `def install` before `virtualenv_install_with_resources`:
```ruby
    ENV["CFLAGS"] = "-I#{Formula["portaudio"].opt_include}"
    ENV["LDFLAGS"] = "-L#{Formula["portaudio"].opt_lib}"
```
Re-run Step 1, then this step.

- [ ] **Step 4: Run the formula's own test block**

Run: `brew test --verbose ./Formula/aai.rb`
Expected: PASS.

- [ ] **Step 5: Strict audit**

Run: `brew audit --strict --online --formula ./Formula/aai.rb 2>&1 | tail -20`
Expected: no errors. `--online` validates the URLs/licenses. Address any "audit" (not "style") errors it lists.

- [ ] **Step 6: Clean uninstall to verify no leftovers**

Run: `brew uninstall aai && echo "uninstalled clean"`
Expected: "uninstalled clean". Confirms the formula owns its full lifecycle (unlike a PyApp-style runtime cache).

- [ ] **Step 7: Commit any dep/install changes**

```bash
git add Formula/aai.rb
git commit -m "build(brew): finalize build deps and portaudio linkage"
```

---

## Task 5: Document the tap install path

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a Homebrew section to README**

Insert under the existing install instructions:

```markdown
## Install with Homebrew

```sh
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew install aai
```

Upgrade with `brew upgrade aai`; remove with `brew uninstall aai`.
```

- [ ] **Step 2: Verify the documented commands work from a clean tap**

```bash
brew untap assemblyai/cli 2>/dev/null || true
brew tap assemblyai/cli https://github.com/AssemblyAI/cli
brew install aai
aai version
```
Expected: installs from the tapped repo and prints the version. (Requires Tasks 2–4 committed and pushed.)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document Homebrew install via the assemblyai/cli tap"
```

---

## Task 6: Release-on-merge pipeline (python-semantic-release + guarded formula bump)

Every merge to `main` that contains a `feat:`/`fix:`/`perf:` (or `BREAKING CHANGE`) commit cuts a new SemVer release; `chore:`/`docs:`/`ci:`/`test:`/`build:`-only merges release nothing (by design — each release rebuilds the tap). Versions are derived from Conventional Commits, which your history already follows. No PyPI publish occurs.

**Files:**
- Modify: `pyproject.toml` (add `[tool.semantic_release]`)
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Add semantic-release config to pyproject.toml**

Append to `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
commit_parser = "conventional"
build_command = ""            # no wheel/sdist build — the formula builds from the tag tarball
tag_format = "v{version}"
allow_zero_version = true     # stay in 0.x
major_on_zero = false         # a breaking change bumps 0.x minor, not 1.0, until you opt in

[tool.semantic_release.branches.main]
match = "main"
```

- [ ] **Step 2: Verify the config parses and resolves the version**

Run: `python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(d['tool']['semantic_release']['version_toml'])"`
Expected: `['pyproject.toml:project.version']` (confirms valid TOML and the key exists).

- [ ] **Step 3: Create the release workflow**

Create `.github/workflows/release.yml`:

```yaml
name: release
on:
  push:
    branches: [main]
    paths-ignore:        # the bot's own formula commit must not re-trigger us
      - "Formula/**"
      - "docs/**"
      - "**/*.md"
concurrency:
  group: release
  cancel-in-progress: false
permissions:
  contents: write
jobs:
  release:
    runs-on: ubuntu-latest
    if: "!contains(github.event.head_commit.message, '[skip ci]')"
    outputs:
      released: ${{ steps.psr.outputs.released }}
      version: ${{ steps.psr.outputs.version }}
      tag: ${{ steps.psr.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      - name: Python Semantic Release
        id: psr
        uses: python-semantic-release/python-semantic-release@v9
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}

  bump-formula:
    needs: release
    if: needs.release.outputs.released == 'true'
    runs-on: macos-latest      # brew + BSD sed preinstalled
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
          fetch-depth: 0
          token: ${{ secrets.GITHUB_TOKEN }}
      - name: Compute url + sha256 for the new tag
        id: meta
        run: |
          TAG="${{ needs.release.outputs.tag }}"
          URL="https://github.com/${GITHUB_REPOSITORY}/archive/refs/tags/${TAG}.tar.gz"
          SHA=$(curl -fsSL "$URL" | shasum -a 256 | cut -d' ' -f1)
          echo "url=$URL" >> "$GITHUB_OUTPUT"
          echo "sha=$SHA" >> "$GITHUB_OUTPUT"
      - name: Rewrite formula url + sha256
        run: |
          sed -i '' -E "s|^  url \".*\"|  url \"${{ steps.meta.outputs.url }}\"|" Formula/aai.rb
          sed -i '' -E "s|^  sha256 \".*\"|  sha256 \"${{ steps.meta.outputs.sha }}\"|" Formula/aai.rb
      - name: Refresh resource stanzas
        run: brew update-python-resources Formula/aai.rb
      - name: Commit formula back to main
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add Formula/aai.rb
          git commit -m "build(brew): aai ${{ needs.release.outputs.version }} [skip ci]" \
            || { echo "no formula changes"; exit 0; }
          git push origin main
```

- [ ] **Step 4: Validate workflow YAML parses**

Run: `brew install yq 2>/dev/null; yq '.jobs | keys' .github/workflows/release.yml`
Expected: `- release` and `- bump-formula`. (Confirms both jobs parse.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/release.yml
git commit -m "ci: release on merge to main via semantic-release + brew bump"
```

> **Loop protection (three layers, already wired above):** (1) `paths-ignore: [Formula/**]` so the formula commit can't trigger `release`; (2) `[skip ci]` in the bot commit message + the `if: !contains(...)` guard; (3) the bot commit type is `build(brew):`, which the conventional parser never treats as releasable. Any one of these alone prevents the infinite loop.
>
> **Branch-protection caveat:** if `main` is protected, `GITHUB_TOKEN` may be blocked from pushing the version bump (Step "release") and the formula commit (Step "bump-formula"). Either add `github-actions[bot]` to the bypass allowlist, or supply a fine-grained PAT as a secret and use it as the `token:`/`github_token:`. Verify after first run that both pushes land on `main`.

---

## Appendix A — Offline resource generator from uv.lock (FALLBACK only)

Use only if `brew update-python-resources` cannot resolve. **Caveat:** this emits the *exact* locked versions but does NOT evaluate platform markers — you must manually drop `pywin32-ctypes` and wrap `jeepney`/`secretstorage` in `on_linux` (see Task 3, Step 3).

Create `scripts/generate_brew_resources.py`:

```python
"""Emit Homebrew resource stanzas for aai-cli's runtime closure from uv.lock.
FALLBACK ONLY: does not evaluate platform markers. See plan Task 3 Step 3."""
from __future__ import annotations

import tomllib
from pathlib import Path

lock = tomllib.loads(Path("uv.lock").read_text())
pkgs = {p["name"]: p for p in lock["package"]}

root = pkgs["aai-cli"]
queue = [d["name"] for d in root.get("dependencies", [])]
seen: set[str] = set()
while queue:
    name = queue.pop()
    if name in seen:
        continue
    seen.add(name)
    queue.extend(d["name"] for d in pkgs.get(name, {}).get("dependencies", []))
seen.discard("aai-cli")

# Windows-only / drop; Linux-only / caller must wrap in on_linux:
WINDOWS_ONLY = {"pywin32-ctypes"}

for name in sorted(seen):
    if name in WINDOWS_ONLY:
        continue
    sdist = pkgs[name].get("sdist")
    if not sdist:
        print(f"# WARNING: {name} has no sdist (wheel-only) — handle manually")
        continue
    digest = sdist["hash"].removeprefix("sha256:")
    print(f'  resource "{name}" do')
    print(f'    url "{sdist["url"]}"')
    print(f'    sha256 "{digest}"')
    print("  end")
    print()
```

Run: `python3 scripts/generate_brew_resources.py > /tmp/aai-resources.rb`
Then proceed from Task 3, Step 2, and apply the Step 3 platform reconciliation manually.

---

## Appendix B — cryptography build optimization

If `cryptography` building from sdist (Rust + OpenSSL) is too slow/fragile, use Homebrew's prebuilt instead:

1. In `Formula/aai.rb`, replace `depends_on "openssl@3"` region with:
   ```ruby
   depends_on "certifi"
   depends_on "cryptography" => :no_linkage
   ```
2. Remove the `resource "cryptography"` block (it now comes from the brew formula).
3. Re-run Task 4, Step 1.

This mirrors what `glances` and `datasette` do.

---

## Self-Review

- **Spec coverage:** "create the tap in this repo" → Task 1 (formula in `Formula/`), Task 5 Step 2 (`brew tap <url>`). "virtualenv way" → `include Language::Python::Virtualenv` + resources (Tasks 1, 3). "no PyPI for our package" → `url` is the GitHub tag tarball (Task 1/2); only third-party deps reference PyPI (Task 3). Native deps (`portaudio`) → Task 1/4. "every merge to main releases a new version" → Task 6 (`python-semantic-release` on merge to `main` + auto formula bump; Conventional Commits → SemVer, so user-facing merges release). Name collision + audio risk → Decisions & Risks, Task 4 Step 3. Loop/branch-protection risks → Task 6 callouts.
- **Placeholder scan:** The only intentional placeholder is the dummy `sha256` in Task 1, explicitly resolved in Task 2 Step 3. No "TBD"/"add error handling"/"similar to" left in steps.
- **Type/name consistency:** Formula class `Aai`, file `Formula/aai.rb`, binary `aai`, test uses `aai version` consistently across Tasks 1, 4, 5. Resource generator references `uv.lock` keys (`package`, `dependencies`, `sdist.url`, `sdist.hash`) confirmed present in the actual lockfile.
