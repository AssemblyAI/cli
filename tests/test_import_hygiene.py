"""The CLI's import chain must stay free of the heavy eval scoring stack.

`assembly eval`'s WER scoring pulls in jiwer (and its rapidfuzz backend). That
import is lazy by design (`wer.py`), so every other command keeps working on an
install that doesn't ship the scoring stack — a module-scope `import jiwer` once
crashed *every* invocation (even `--help`) of a Homebrew install whose formula
lacked those resources.
"""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import sys
import aai_cli.main
heavy = {"jiwer", "rapidfuzz"}
loaded = sorted(heavy & {name.split(".")[0] for name in sys.modules})
assert not loaded, f"CLI import eagerly loaded: {loaded}"
"""


def test_cli_import_does_not_load_eval_scoring_stack():
    # A fresh interpreter: the suite's own process has long since imported the
    # scoring stack, so sys.modules can only be probed in a child.
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
