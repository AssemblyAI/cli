---
name: check
description: Run the full local verification gate for the assembly CLI (the same checks CI runs). Use before pushing or opening a PR.
disable-model-invocation: true
---

# check

Run the project's canonical verification gate (`./scripts/check.sh`) and report the result.

## Steps

1. Run the full gate:

   ```sh
   ./scripts/check.sh
   ```

   Everything Python runs through `uv run` against the locked environment. Do not claim
   success until the script prints `All checks passed.`

2. If anything fails, fix it and re-run `./scripts/check.sh` until it passes.

## Gate stages (mirrors scripts/check.sh)

`scripts/check.sh` is the single source of truth. The list below is generated from its
`==>` stage labels by `scripts/check_stages_gate.py`, and a gate step keeps the two in
sync — so it is always current (run `uv run python scripts/check_stages_gate.py --write`
after adding or reordering a stage):

<!-- BEGIN GATE STAGES (generated from scripts/check.sh by scripts/check_stages_gate.py --write; do not edit by hand) -->

1. uv lock freshness
2. validate-pyproject (pyproject.toml schema)
3. ruff check (src + tests)
4. ruff format --check (src + tests)
5. mypy (src + tests)
6. pyright (src strict)
7. pyright (tests standard)
8. vulture (dead-code gate, src + tests)
9. deptry (dependency hygiene)
10. import-linter (architecture contracts)
11. max file length (500-line gate, src + tests + scripts)
12. xenon (cyclomatic complexity gate, src only)
13. swiftlint (macOS audio helper)
14. swift compile (macOS audio helper)
15. markdownlint (docs/ is generated, so excluded)
16. codespell (spell-check code, comments, docs)
17. json validity (all tracked + staged *.json)
18. prettier (init template JS/CSS)
19. shellcheck
20. actionlint (GitHub Actions workflow lint)
21. zizmor (GitHub Actions security audit)
22. gitleaks (secret scan)
23. generated --show-code compile gate
24. init template contract/import gate
25. unused snapshot/fixture gate
26. docs consistency gate (env vars / exit codes / command refs)
27. docstring coverage gate (public API ratchet)
28. gate-stage docs sync (the /check skill mirrors this script)
29. brew audit (Homebrew formula)
30. pytest (with branch-coverage gate)
31. diff-cover (patch coverage: every changed line must be tested)
32. mutation gate (diff-scoped: a changed line's test must fail when it breaks)
33. no new static-analysis escape hatches
34. codeql (security + quality suites, mirrors codeql.yml minus swift)
35. build + twine check (PyPI publish readiness)

<!-- END GATE STAGES -->

The non-`ruff`/`mypy` stages that most often surprise an otherwise-clean change: `vulture`
(unused code), `deptry` (dependency hygiene), `lint-imports` (architecture contracts),
`xenon` (cyclomatic complexity > grade B), `codespell`, and the docs-consistency gate
(REFERENCE.md/README.md must name only real commands, env vars, and exit codes).

## Iterate fast, then gate once

The full gate is slow (codeql, the ~2.5k-test suite, and the build), and the pre-commit
hook re-requires a green run after *any* edit — so don't loop on the whole script.
Iterate with the targeted commands, then run `./scripts/check.sh` once at the end:

```sh
uv run ruff check . && uv run mypy && uv run pyright   # the static trio
uv run pytest tests/test_foo.py -q                     # just the file(s) you touched
uv run pytest --snapshot-update                        # after any help/output change
```

## The stages that fail last (and how to re-run them alone)

`diff-cover` (100% patch coverage vs `origin/main`) and the mutation gate run near the
end, so a weak or missing test is discovered late. After a gate run (or any pytest run
with the coverage flags), re-run just those two:

```sh
uv run pytest -q -n auto --cov=aai_cli --cov-branch --cov-context=test --cov-report=xml
uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100
uv run python scripts/mutation_gate.py origin/main
```

## Optional, opt-in suites (not run by check.sh)

Run these only when relevant — they are slow and/or need credentials:

```sh
uv run pytest -m e2e             # real-API end-to-end; needs ASSEMBLYAI_API_KEY, else skips
uv run pytest -m install         # installs each init template's requirements; needs network + uv
```

## Notes

- External tools that aren't Python deps — `shellcheck`, `prettier`, `swiftlint`/`swiftc`,
  `actionlint`, `gitleaks`, `brew`, `codeql` — self-skip with a notice when not installed
  (CI and the web session-start hook still run them); that's expected, not a failure.
  `swiftlint`/`swiftc` also no-op off macOS.
- `diff-cover`, the mutation gate, and the escape-hatch gate self-skip when `origin/main`
  isn't present (e.g. a shallow branch-only clone); CI provides the base ref.
- Report the final outcome with the actual tail of the output, not a summary from memory.
