"""Compatibility shim — model.py has moved to aai_cli.agent_cascade.model.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.model import (  # noqa: F401
    build_model,
)
