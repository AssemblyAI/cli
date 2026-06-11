#!/usr/bin/env python3
"""Gate: CLI startup must not grow new third-party imports.

Every `aai` invocation — including `aai --help` — pays for whatever
`aai_cli.main` transitively imports before the first byte of output. A
top-level `import openai` added three modules away from main.py slows every
command down, passes every test, and is invisible to the other gates. This
gate pins the startup import graph instead of wall-clock time, so it is
deterministic across machines.

It spawns a fresh interpreter, imports `aai_cli.main` (what the `aai` console
script loads), and fails if any third-party top-level module outside
ALLOWED_STARTUP_MODULES appears in `sys.modules`. The baseline is a ratchet
like check.sh's `Any`/`cast(` count gates: it encodes the status quo, new
entries require editing this file deliberately, and shrinking it (by making an
import lazy) is always welcome — the gate prints baseline entries that are no
longer imported so they can be removed.
"""

from __future__ import annotations

import subprocess
import sys

# The committed contract: every third-party top-level module the CLI currently
# imports at startup. Anything not listed here must be imported lazily, inside
# the command body that needs it. Notably absent (and intended to stay that
# way): openai, sounddevice, yt_dlp, questionary.
ALLOWED_STARTUP_MODULES = frozenset(
    {
        # CLI framework and terminal rendering — inherent to a Typer app.
        "typer",
        "click",
        "rich",
        "pygments",
        "markdown_it",
        "mdurl",
        "shellingham",
        # Config/profile persistence and the OS keyring.
        "platformdirs",
        "tomli_w",
        "keyring",
        "jaraco",  # keyring dependency
        "more_itertools",  # keyring dependency
        # Heavy at startup today; candidates for lazy imports. Remove an entry
        # here once its import moves into the command that uses it.
        "assemblyai",
        "httpx",  # transitive via the assemblyai SDK
        "httpx2",
        "idna",
        "websockets",
        "pydantic",
        "pydantic_core",
        "annotated_types",  # pydantic dependency
        "annotated_doc",  # pydantic dependency
        "typing_extensions",
        "typing_inspection",  # pydantic dependency
        # stdlib on 3.12, audioop-lts backport (third-party) on 3.13+.
        "audioop",
    }
)

# Present in sys.modules from how the interpreter was launched, not from our
# import graph: venv/site injection and the probe's own -c entry point.
NOT_DEPENDENCIES = frozenset({"sitecustomize", "usercustomize", "_virtualenv", "__main__"})

PROBE = "import aai_cli.main, sys; print(chr(10).join(sys.modules))"


def startup_third_party_modules() -> set[str]:
    """Top-level third-party modules loaded by `import aai_cli.main` in a fresh process."""
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        msg = "startup probe failed: `import aai_cli.main` errored in a fresh interpreter"
        raise RuntimeError(msg)
    top_level = {name.partition(".")[0] for name in proc.stdout.split()}
    return {
        name
        for name in top_level
        if name != "aai_cli"
        and name not in sys.stdlib_module_names
        and name not in NOT_DEPENDENCIES
        and not name.startswith("_sysconfigdata")  # platform-specific generated stdlib module
    }


def main() -> int:
    imported = startup_third_party_modules()

    offenders = sorted(imported - ALLOWED_STARTUP_MODULES)
    if offenders:
        sys.stdout.write("New third-party module(s) imported at CLI startup:\n")
        for name in offenders:
            sys.stdout.write(f"  {name}\n")
        sys.stdout.write(
            "Import them lazily inside the command that needs them, or add them to\n"
            "ALLOWED_STARTUP_MODULES in scripts/startup_import_gate.py deliberately.\n"
        )
        return 1

    stale = sorted(ALLOWED_STARTUP_MODULES - imported - frozenset(sys.stdlib_module_names))
    if stale:
        sys.stdout.write(
            f"   note: no longer imported at startup, removable from baseline: {stale}\n"
        )

    sys.stdout.write(
        f"   startup imports {len(imported)} third-party modules, all within the baseline\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
