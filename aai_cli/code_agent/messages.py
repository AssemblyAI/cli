"""Compatibility shim — messages.py has moved to aai_cli.agent_cascade.messages.

This re-export keeps the ``assembly code`` command working until it is removed in
the next task. Do not add new imports here.
"""

from __future__ import annotations

from aai_cli.agent_cascade.messages import (  # noqa: F401
    AssistantMessage,
    ErrorMessage,
    Note,
    ToolCallLine,
    ToolOutput,
    UserMessage,
)
