"""Compatibility shim — summarize.py has moved to aai_cli.agent_cascade.summarize.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.summarize import (  # noqa: F401
    describe_args,
    full_args,
    summarize_call,
    summarize_result,
)
