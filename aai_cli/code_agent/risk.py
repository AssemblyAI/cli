"""Compatibility shim — risk.py has moved to aai_cli.agent_cascade.risk.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.risk import (  # noqa: F401
    FETCH_TOOL_NAME,
    risk_warning,
)
