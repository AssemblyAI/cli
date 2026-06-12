"""Sniffing the raw, not-yet-parsed command line for output-mode flags.

Both the root callback (`main`) and telemetry's first-run notice run before any
subcommand parses its own ``--json``, so honoring a pipeline's request for
machine-readable output at that point means scanning the raw token list. The
shared definition lives here — free of Rich and import cycles — so the two
callers can't drift on which flag forms count.
"""

from __future__ import annotations


def requests_json(raw_args: list[str]) -> bool:
    """Whether the token list opts into JSON output: ``--json``, ``-j``,
    ``-o json``, ``--output json``, or their glued forms (``--output=json``,
    ``-ojson``)."""
    for index, token in enumerate(raw_args):
        if token in ("--json", "-j", "--output=json", "-ojson"):
            return True
        if token in ("-o", "--output") and raw_args[index + 1 : index + 2] == ["json"]:
            return True
    return False
