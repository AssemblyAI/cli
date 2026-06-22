"""Compatibility shim — banner.py has moved to aai_cli.agent_cascade.banner.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.banner import (  # noqa: F401
    BRAND_HEX,
    READY_LINE,
    TIP_LINE,
    version,
    wordmark,
)
