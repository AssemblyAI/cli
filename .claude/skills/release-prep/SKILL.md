---
name: release-prep
description: Prepare an aai CLI release — bump the version, run the full gate, build and validate the wheel/sdist, and smoke-test the real install. Use when cutting a new release.
disable-model-invocation: true
---

# release-prep

Drive an `aai` release to a verified, publishable state. Stop and report at the first failure — never push or publish on a red check.

## 1. Version bump

- Update `version` in `pyproject.toml` (`[project]`). Confirm `aai_cli/__init__.py` `__version__` stays in sync (the `version` command and install smoke test read it).
- Decide the bump (patch/minor/major) from what changed since the last tag; ask the user if it's ambiguous.

## 2. Full gate

```sh
./scripts/check.sh
```

Must end with `All checks passed.` (ruff, mypy, markdownlint, shellcheck, pytest+coverage, build, `twine check --strict`).

## 3. Real install smoke test

```sh
uv run pytest -q -m install_script
```

This builds the wheel and runs `install.sh` for real (pipx + pip --user), asserting `aai` runs. Needs network + uv/pipx.

## 4. Build + metadata validation

```sh
rm -rf dist && uv build && uvx twine check --strict dist/*
```

Confirm both an sdist and a wheel are produced and the README renders for PyPI.

## Distribution caveat

The PyPI name **`assemblyai-cli` is squatted by a third party** — do **not** assume `pip install assemblyai-cli` resolves to this project. Publishing/distribution currently goes through `install.sh` (git install via pipx / pip --user) and any Homebrew tap, not that PyPI name. Flag this if a release step assumes the squatted name.

## Output

Report the version bumped to, the gate result (with output tail), and confirm `dist/` contains a validated wheel + sdist. Only then is the release ready to tag/push.
