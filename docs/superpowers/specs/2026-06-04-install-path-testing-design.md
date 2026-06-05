# Install-path testing design

## Problem

CI never exercises the public install story. All four `ci.yml` jobs install the
package with `pip install -e .` (editable, from the local checkout). The path a
real user takes — `curl -fsSL .../install.sh | sh` → `pipx install git+https://…`
→ `aai` on PATH — is untested. A regression that breaks it (a bad dependency
version floor, a missing/renamed console entrypoint, an `install.sh` bug, a PATH
problem) passes CI green.

`install.sh` itself contains untested logic: a Python 3.10+ gate, construction of
the `git+https://github.com/AssemblyAI/cli.git@REF` spec from `AAI_REPO`/`AAI_REF`,
a pipx-vs-`pip --user` fallback branch, and PATH-check messaging.

## Goal

Catch install-path regressions before release, in layers that trade speed for
fidelity, without making every PR slow or flaky.

## Non-goals

- Testing install from a *pushed* git ref / the literal documented `curl | sh`
  against GitHub. We install the PR's own code from a local wheel instead (see
  the `AAI_SPEC` seam). A pushed-ref / nightly variant is explicitly out of scope
  for this work.
- Publishing to or installing from PyPI (the `assemblyai-cli` name is squatted;
  `install.sh` uses the GitHub git spec, so PyPI is irrelevant here).

## Design

Three layers plus one small change to `install.sh`.

### Layer 0 — `install.sh` testability seam

Add an `AAI_SPEC` environment override. When set, `install.sh` installs that
spec verbatim instead of constructing the `git+https://…@REF` URL. Documented
in the script as test-only. This is the hook that lets Layers 2 and 3 install
the PR's actual code without pushing a commit.

Precedence: `AAI_SPEC` (if set) wins; otherwise build the spec from
`AAI_REPO`/`AAI_REF` as today. The rest of the script (Python gate, pipx/pip
branch, PATH check) is unchanged.

### Layer 1 — Fast shell-logic unit tests (every PR)

New `tests/test_install_sh.py`. Runs `install.sh` via `subprocess` with a
sandboxed `PATH` containing fake shims:

- a recording `pipx` shim that writes its argv to a file and exits 0
- a `python3` shim that reports a chosen version (to drive the gate)
- a `pip` shim (recording) for the fallback branch

Cases:

1. No `python3`/`python` on PATH → exit 1, version error on stderr.
2. Python < 3.10 → exit 1, version error.
3. pipx present → invokes `pipx install --force <spec>`.
4. pipx absent → invokes `python -m pip install --user --upgrade <spec>`.
5. Default → spec is `git+https://github.com/AssemblyAI/cli.git@main`.
6. `AAI_REPO`/`AAI_REF` set → spec string reflects them.
7. `AAI_SPEC` set → used verbatim, no git URL constructed.
8. PATH check: `aai` present → "Installed. Next: …"; absent → ensurepath hint.

No network; runs in milliseconds in the default suite (the existing `check` job).

Also add `shellcheck install.sh` as a static gate in `scripts/check.sh` (and
therefore CI), guarded so it's skipped with a notice when `shellcheck` isn't
installed locally.

### Layer 2 — Real-install smoke test (new marker)

New pytest test marked `install_script` (a new marker, kept separate from the
existing `install` marker used by `test_init_template_install.py`, so the new
CI job stays tight and the template-install test remains manual-only).

The test:

1. Builds a wheel from the checkout (`uv build` / `python -m build`) into a temp dir.
2. Runs `install.sh` with `AAI_SPEC=<path-to-wheel>` (plus any extra index/deps
   needed so the wheel's dependencies resolve from PyPI).
3. Asserts the installed `aai` runs: `aai --version` exits 0 and prints the
   package version.

Parametrized over the install branch:

- `pipx` available → pipx path.
- `pipx` hidden from PATH → `pip --user` fallback path.

Only *dependencies* hit the network (the package itself is the local wheel), so
each install is ~30–90s rather than a full git build.

### Layer 3 — `install-smoke` CI job

A new job in `ci.yml` that runs `pytest -m install_script` on a matrix:

| OS | pipx path | pip --user fallback |
|----|-----------|---------------------|
| ubuntu-latest | ✅ | ✅ |
| macos-latest | ✅ | ❌ (excluded) |

Rationale for the macOS exclusion: the pipx-vs-pip branch is OS-independent
shell logic, proven once on Linux. macOS's distinct risk is environmental
(Homebrew Python, `~/Library` paths, PEP 668 "externally-managed-environment").
The **pipx** path is the documented primary path and sidesteps PEP 668; the
`pip --user` path on macOS is the most likely to fail for reasons outside the
script's control, so gating PRs on it would add flakiness without proportional
signal.

The job pins actions to commit SHAs and installs `uv`/build tooling consistent
with the existing jobs.

## Error handling and skips

Mirror the `test_init_template_install.py` convention: the Layer 2 test **skips
(never fails)** when the machine can't run it — offline (PyPI unreachable) or
`uv`/build tooling absent. Keeps keyless/offline/sandboxed local runs unblocked;
CI has the tooling and network, so it runs for real there.

## Testing the tests

- Layer 1 cases are deterministic (fake shims, fixed inputs), no network.
- Layer 2 is the smoke test itself; its skip logic is the only branch worth a
  quick sanity check (simulate offline → assert skipped, not failed).

## Files touched

- `install.sh` — add `AAI_SPEC` override + doc comment.
- `tests/test_install_sh.py` — new (Layer 1).
- `tests/test_install_script_smoke.py` — new (Layer 2), marked `install_script`.
- `pyproject.toml` — register the `install_script` marker.
- `scripts/check.sh` — add `shellcheck install.sh` (guarded).
- `.github/workflows/ci.yml` — add the `install-smoke` job.
