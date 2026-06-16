#!/usr/bin/env python3
"""Prototype Python orchestrator for the project gate — a ``scripts/check.sh`` alternative.

A sketch for evaluation *alongside* the canonical bash gate; deliberately NOT yet wired
into CI or the commit hook. It declares the same stages as data, which buys ergonomics
the linear shell script can't easily offer: run one stage or a subset
(``check.py mypy ruff-check``), list them (``--list``), run the independent
static-analysis stages in parallel and report *every* failure at once
(bash ``set -e`` stops at the first; ``--no-parallel`` forces sequential), and a
per-stage timing + PASS/FAIL summary.

The load-bearing semantic from check.sh is preserved exactly: the gate marker
(``scripts/gate_marker.py record``) is written ONLY after a full, unfiltered run in which
every stage passed, so it can never let a ``git commit`` through on a partial run. The
coverage-dependent tail (pytest -> diff-cover -> mutation) and the artifact build run
serially after the parallel batch, since their ordering and file outputs are real.

stdlib-only, like gate_marker.py, so the gate needs no project import to bootstrap.
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parent.parent
PKG = "aai_cli"
SWIFT = "aai_cli/streaming/macos_system_audio.swift"

# A stage writes its tool output into the given stream and returns True on success.
Runner = Callable[[TextIO], bool]

# Net-new escape hatches are count-gated against the merge-base, same policy as check.sh.
# (label, PCRE pattern, paths) — scoped to shipped code + tests, never the dev scripts.
HATCH_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ignore/no-cover", r"# type: ignore|# noqa|pragma: no cover", (PKG, "tests")),
    (
        "test skip/xfail/sleep",
        r"pytest\.skip\(|pytest\.xfail\(|@pytest\.mark\.(skip|xfail)|\btime\.sleep\(",
        ("tests",),
    ),
    ("Any", r"Any", (PKG, "tests")),
    ("cast()", r"cast\(", (PKG, "tests")),
)


def _capture(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run *argv* from the repo root, capturing output; never raises on non-zero exit."""
    return subprocess.run(list(argv), cwd=ROOT, capture_output=True, text=True, check=False)


def _run(argv: Sequence[str], out: TextIO) -> bool:
    """Run *argv* from the repo root, tee combined stdout+stderr into *out*, return success."""
    proc = subprocess.Popen(
        list(argv),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout is not None:
        for line in proc.stdout:
            out.write(line)
    return proc.wait() == 0


def cmd(line: str) -> Runner:
    """Build a stage runner from a shell-style command line (split with shlex, no shell)."""
    argv = shlex.split(line)

    def run(out: TextIO) -> bool:
        return _run(argv, out)

    return run


def optional(tool: str, runner: Runner, skip_msg: str) -> Runner:
    """Wrap *runner* so it self-skips (and passes) when *tool* isn't on PATH."""

    def run(out: TextIO) -> bool:
        if shutil.which(tool) is None:
            out.write(f"   {skip_msg}\n")
            return True
        return runner(out)

    return run


def _has_origin_main() -> bool:
    return _capture(["git", "rev-parse", "--verify", "--quiet", "origin/main"]).returncode == 0


def diff_gated(runner: Runner, label: str) -> Runner:
    """Wrap *runner* so it self-skips when origin/main is absent (shallow clone)."""

    def run(out: TextIO) -> bool:
        if not _has_origin_main():
            out.write(f"   origin/main not found; skipping {label} (CI provides it)\n")
            return True
        return runner(out)

    return run


def _showcode(out: TextIO) -> bool:
    tmp = Path(tempfile.mkdtemp())
    try:
        if not _run(
            shlex.split(f"uv run python scripts/generated_code_compile_gate.py {tmp}"), out
        ):
            return False
        return _run(shlex.split(f"uv run python -m compileall -q {tmp}"), out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _build(out: TextIO) -> bool:
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    if not _run(["uv", "build"], out):
        return False
    dists = sorted(str(p) for p in (ROOT / "dist").glob("*"))
    if not dists:
        out.write("   no build artifacts produced\n")
        return False
    return _run(["uvx", "twine", "check", "--strict", *dists], out)


def _swiftlint(out: TextIO) -> bool:
    if shutil.which("swiftlint") is None:
        out.write("   swiftlint not found; skipping (install with: brew install swiftlint)\n")
        return True
    return _run(["swiftlint", "lint", "--no-cache", "--strict", SWIFT], out)


def _swift_compile(out: TextIO) -> bool:
    if sys.platform != "darwin":
        out.write("   not macOS; skipping compile for macOS-only frameworks\n")
        return True
    if shutil.which("swiftc") is None:
        out.write("   swiftc not found; skipping (macOS system audio builds on first use)\n")
        return True
    cache = Path(tempfile.mkdtemp())
    helper = cache / "aai-macos-audio-check"
    frameworks = "-framework ScreenCaptureKit -framework AVFoundation -framework CoreMedia -framework CoreGraphics"
    try:
        build = f"swiftc -parse-as-library {SWIFT} -module-cache-path {cache} -O {frameworks} -o {helper}"
        if not _run(shlex.split(build), out):
            return False
        probe = _capture([str(helper), "--unknown-check-flag"])
        out.write(probe.stdout)
        out.write(probe.stderr)
        if probe.returncode == 0:
            out.write("   expected Swift helper argument validation to fail\n")
            return False
        return "Unknown argument: --unknown-check-flag" in probe.stderr
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def _brew_audit(out: TextIO) -> bool:
    if shutil.which("brew") is None:
        out.write("   brew not found; skipping (Homebrew CI / release runner has it)\n")
        return True
    repo = _capture(["brew", "--repository"]).stdout.strip()
    tap = Path(repo) / "Library" / "Taps" / "local" / "homebrew-aaiaudit"
    (tap / "Formula").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "Formula" / "assembly.rb", tap / "Formula" / "assembly.rb")
    try:
        return _run(["brew", "audit", "--strict", "--formula", "local/aaiaudit/assembly"], out)
    finally:
        shutil.rmtree(tap, ignore_errors=True)


def _grep_count(args: Sequence[str], paths: Sequence[str]) -> int:
    return _capture(["git", "grep", *args, "--", *paths]).stdout.count("\n")


def _escape_hatches(out: TextIO) -> bool:
    base = _capture(["git", "merge-base", "origin/main", "HEAD"])
    gate_base = base.stdout.strip() or "origin/main"
    ok = True
    for label, pattern, paths in HATCH_GROUPS:
        base_n = _grep_count(["-hP", pattern, gate_base], paths)
        work_n = _grep_count(["--untracked", "-hP", pattern], paths)
        if work_n > base_n:
            out.write(
                f"New {label} escape hatch: {work_n} current vs {base_n} at the merge-base. "
                "Refactor it or update the gate explicitly.\n"
            )
            ok = False
    return ok


@dataclass
class Stage:
    """One gate step: a short selectable *key*, a human *title*, and how to *run* it."""

    key: str
    title: str
    run: Runner
    parallel: bool = True


@dataclass
class Result:
    """The outcome of running one stage."""

    stage: Stage
    ok: bool
    seconds: float
    output: str


_PYTEST = (
    "uv run pytest -q --strict-config --strict-markers -n auto -m 'not e2e and not install' "
    "--cov=aai_cli --cov-branch --cov-context=test --cov-report=term-missing "
    "--cov-report=xml --cov-fail-under=90"
)
_SHELLCHECK = (
    "shellcheck -x --source-path=. scripts/check.sh scripts/docker_build_check.sh "
    "scripts/cut_release.sh scripts/gate_tool_pins.sh .claude/hooks/session-start.sh "
    ".claude/hooks/require-gate-before-commit.sh"
)


def _stages() -> list[Stage]:
    return [
        Stage("uv-lock", "uv lock freshness", cmd("uv lock --check")),
        Stage("ruff-check", "ruff check (src + tests)", cmd("uv run ruff check .")),
        Stage("ruff-format", "ruff format --check", cmd("uv run ruff format --check .")),
        Stage("mypy", "mypy (src + tests)", cmd("uv run mypy")),
        Stage("pyright-src", "pyright (src strict)", cmd("uv run pyright")),
        Stage(
            "pyright-tests", "pyright (tests)", cmd("uv run pyright -p pyrightconfig.tests.json")
        ),
        Stage("vulture", "vulture (dead-code gate)", cmd("uv run vulture")),
        Stage("deptry", "deptry (dependency hygiene)", cmd("uv run deptry .")),
        Stage("import-linter", "import-linter (architecture)", cmd("uv run lint-imports")),
        Stage(
            "max-file-length",
            "max file length (500-line gate)",
            cmd("uv run python scripts/max_file_length.py"),
        ),
        Stage(
            "xenon",
            "xenon (complexity gate)",
            cmd(f"uv run xenon --max-absolute B --max-modules A --max-average A {PKG}"),
        ),
        Stage("swiftlint", "swiftlint (macOS audio helper)", _swiftlint),
        Stage("swift-compile", "swift compile (macOS audio helper)", _swift_compile),
        Stage(
            "markdownlint",
            "markdownlint",
            optional(
                "markdownlint",
                cmd(
                    "markdownlint **/*.md --ignore docs --ignore node_modules --ignore .pytest_cache"
                ),
                "markdownlint not found; skipping (CI runs it)",
            ),
        ),
        Stage(
            "codespell",
            "codespell (spell-check)",
            optional(
                "uvx",
                cmd("uvx codespell ."),
                "uvx not found; skipping (pre-commit + CI run codespell)",
            ),
        ),
        Stage("json-lint", "json validity", cmd("uv run python scripts/json_lint.py")),
        Stage(
            "prettier",
            "prettier (init template JS/CSS)",
            optional(
                "prettier",
                cmd("prettier --check aai_cli/init/templates/**/*.{js,css}"),
                "prettier not found; skipping (CI runs it)",
            ),
        ),
        Stage(
            "shellcheck",
            "shellcheck",
            optional("shellcheck", cmd(_SHELLCHECK), "shellcheck not found; skipping (CI runs it)"),
        ),
        Stage(
            "actionlint",
            "actionlint (workflow lint)",
            optional(
                "actionlint", cmd("actionlint"), "actionlint not found; skipping (CI runs it)"
            ),
        ),
        Stage(
            "zizmor",
            "zizmor (workflow security audit)",
            cmd("uv run zizmor --offline .github/workflows"),
        ),
        Stage(
            "gitleaks",
            "gitleaks (secret scan)",
            optional(
                "gitleaks",
                cmd("gitleaks dir --no-banner --redact -c .gitleaks.toml ."),
                "gitleaks not found; skipping (CI runs it)",
            ),
        ),
        Stage("showcode", "generated --show-code compile gate", _showcode),
        Stage(
            "template-contract",
            "init template contract/import gate",
            cmd("uv run python scripts/template_contract_gate.py"),
        ),
        Stage(
            "unused-fixtures",
            "unused snapshot/fixture gate",
            cmd("uv run python scripts/unused_fixtures_gate.py"),
        ),
        Stage(
            "docs-consistency",
            "docs consistency gate",
            cmd("uv run python scripts/docs_consistency_gate.py"),
        ),
        Stage(
            "docstring-coverage",
            "docstring coverage gate",
            cmd("uv run python scripts/docstring_coverage_gate.py"),
        ),
        Stage("brew-audit", "brew audit (Homebrew formula)", _brew_audit),
        Stage(
            "codeql",
            "codeql (security + quality suites)",
            optional(
                "codeql",
                cmd("uv run python scripts/codeql_gate.py"),
                "codeql not found; skipping (codeql.yml runs it in CI)",
            ),
        ),
        Stage(
            "escape-hatches",
            "no new static-analysis escape hatches",
            diff_gated(_escape_hatches, "escape-hatch diff gate"),
        ),
        # Serial tail: coverage chain must stay ordered, build writes ./dist.
        Stage("pytest", "pytest (branch-coverage gate)", cmd(_PYTEST), parallel=False),
        Stage(
            "diff-cover",
            "diff-cover (patch coverage)",
            diff_gated(
                cmd("uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100"),
                "patch-coverage gate",
            ),
            parallel=False,
        ),
        Stage(
            "mutation",
            "mutation gate",
            diff_gated(cmd("uv run python scripts/mutation_gate.py origin/main"), "mutation gate"),
            parallel=False,
        ),
        Stage("build", "build + twine check", _build, parallel=False),
    ]


def _invoke(stage: Stage, out: TextIO) -> tuple[bool, float]:
    start = time.monotonic()
    try:
        ok = stage.run(out)
    except OSError as exc:
        out.write(f"   orchestrator error: {exc!r}\n")
        ok = False
    return ok, time.monotonic() - start


def _execute_captured(stage: Stage) -> Result:
    buf = io.StringIO()
    ok, seconds = _invoke(stage, buf)
    return Result(stage, ok, seconds, buf.getvalue())


def _print_captured(result: Result) -> None:
    status = "ok" if result.ok else "FAIL"
    sys.stdout.write(f"==> {result.stage.title}  [{status} {result.seconds:.1f}s]\n")
    if result.output:
        sys.stdout.write(result.output)
        if not result.output.endswith("\n"):
            sys.stdout.write("\n")


def _execute_live(stage: Stage) -> Result:
    sys.stdout.write(f"==> {stage.title}\n")
    sys.stdout.flush()
    ok, seconds = _invoke(stage, sys.stdout)
    sys.stdout.write(f"    [{'ok' if ok else 'FAIL'} {seconds:.1f}s]\n")
    return Result(stage, ok, seconds, "")


def _run_parallel(stages: list[Stage]) -> list[Result]:
    if not stages:
        return []
    workers = min(len(stages), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_execute_captured, stages))
    for result in results:
        _print_captured(result)
    return results


def _run_serial(stages: list[Stage]) -> list[Result]:
    results: list[Result] = []
    for stage in stages:
        result = _execute_live(stage)
        results.append(result)
        if not result.ok:
            break  # fail-fast, matching `set -e`
    return results


def _dispatch(selected: list[Stage], *, parallel_mode: bool) -> list[Result]:
    if not parallel_mode:
        return _run_serial(selected)
    concurrent = [s for s in selected if s.parallel]
    sequential = [s for s in selected if not s.parallel]
    results = _run_parallel(concurrent)
    if all(r.ok for r in results):
        results = results + _run_serial(sequential)
    return results


def _record_marker() -> None:
    if _capture(["python3", "scripts/gate_marker.py", "record"]).returncode != 0:
        sys.stdout.write("   (warning: could not record gate marker)\n")


def _finish(results: list[Result], *, total: int, started: float, full_run: bool) -> int:
    elapsed = time.monotonic() - started
    failed = [r for r in results if not r.ok]
    sys.stdout.write(f"\nRan {len(results)}/{total} stage(s) in {elapsed:.1f}s.\n")
    if failed:
        sys.stdout.write("FAILED: " + ", ".join(r.stage.key for r in failed) + "\n")
        return 1
    if len(results) < total:
        sys.stdout.write("Stopped before all stages ran.\n")
        return 1
    if full_run:
        _record_marker()
    sys.stdout.write("All checks passed.\n")
    return 0


def _select(stages: list[Stage], names: list[str]) -> list[Stage]:
    by_key = {s.key: s for s in stages}
    unknown = [n for n in names if n not in by_key]
    if unknown:
        sys.stderr.write(f"unknown stage(s): {', '.join(unknown)}\n")
        sys.stderr.write(f"available: {', '.join(by_key)}\n")
        raise SystemExit(2)
    return [by_key[n] for n in names]


def _print_list(stages: list[Stage]) -> None:
    for stage in stages:
        phase = "parallel" if stage.parallel else "serial"
        sys.stdout.write(f"  {stage.key:18}  {stage.title}  [{phase}]\n")


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project gate (prototype Python orchestrator).")
    parser.add_argument("stages", nargs="*", help="run only these stage keys (default: all)")
    parser.add_argument("--list", action="store_true", help="list stage keys and exit")
    parser.add_argument("--no-parallel", action="store_true", help="run every stage sequentially")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse(argv)
    stages = _stages()
    if args.list:
        _print_list(stages)
        return 0
    selected = _select(stages, args.stages) if args.stages else stages
    parallel_mode = not args.stages and not args.no_parallel
    started = time.monotonic()
    results = _dispatch(selected, parallel_mode=parallel_mode)
    return _finish(results, total=len(selected), started=started, full_run=not args.stages)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
