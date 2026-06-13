"""Sniffing the raw, not-yet-parsed command line for output-mode flags.

Both the root callback (`main`) and telemetry's first-run notice run before any
subcommand parses its own ``--json``, so honoring a pipeline's request for
machine-readable output at that point means scanning the raw token list. The
shared definition lives here — free of Rich and import cycles — so the two
callers can't drift on which flag forms count.
"""

from __future__ import annotations

# The standalone "give me JSON" flag spellings. Shared with the misplaced-flag
# hint (which recognizes a `--json`/`-j` passed at the root level), so the two
# can't drift on which forms count.
JSON_FLAGS = ("--json", "-j")

# Where the root group stashes the raw token list on the Click context before
# parsing (see `_OrderedGroup.parse_args` in main.py). The root callback and the
# Click error formatter (typer_patches.py) both read it to honor a JSON opt-in
# for failures raised before the subcommand parses its own --json.
RAW_ARGS_META_KEY = "aai_raw_args"


def requests_json(raw_args: list[str]) -> bool:
    """Whether the token list opts into JSON output: ``--json``, ``-j``,
    ``-o json``, ``--output json``, or their glued forms (``--output=json``,
    ``-ojson``)."""
    for index, token in enumerate(raw_args):
        if token in (*JSON_FLAGS, "--output=json", "-ojson"):
            return True
        if token in ("-o", "--output") and raw_args[index + 1 : index + 2] == ["json"]:
            return True
    return False


def requests_quiet(raw_args: list[str]) -> bool:
    """Whether the token list asked for quiet output: ``--quiet`` or ``-q``."""
    return any(token in ("--quiet", "-q") for token in raw_args)
