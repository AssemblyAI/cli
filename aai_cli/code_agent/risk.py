"""Heuristic risk flags for tool calls, surfaced on the approval prompt.

The approval modal already shows *what* a tool will do; for the genuinely dangerous calls it
also shows *why to look twice* — a one-line warning, the way deepagents-code badges suspicious
shell commands and URLs. Purely advisory (the real SSRF guard lives in ``fetch_tool``); this
only nudges the human reviewing a manual approval. Pure functions so they unit-test cleanly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from aai_cli.code_agent.fetch_tool import FETCH_TOOL_NAME

# Shell fragments that can destroy data, escalate privileges, or pipe a remote script straight
# into a shell — the classic "are you sure?" cases. Word-ish boundaries avoid matching inside
# innocuous longer tokens (e.g. ``format`` should not trip ``mkfs``).
_DANGEROUS_SHELL = (
    (re.compile(r"\brm\s+(-\w*\s+)*-\w*[rf]", re.I), "deletes files recursively/forcibly"),
    (re.compile(r"\bsudo\b", re.I), "runs with elevated privileges"),
    (re.compile(r"\bmkfs\b|\bdd\s+if=", re.I), "can overwrite a disk or filesystem"),
    (re.compile(r":\s*\(\)\s*\{.*\|.*&\s*\}\s*;"), "looks like a fork bomb"),
    (
        re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b", re.I),
        "pipes a download into a shell",
    ),
    (re.compile(r">\s*/dev/(sd|disk|nvme)", re.I), "writes directly to a block device"),
)
# URL hosts that mean a fetch is reaching a local/internal target rather than the public web.
_LOCAL_HOST = re.compile(
    r"^(localhost|127\.|0\.0\.0\.0|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1\]?)",
    re.I,
)


def _shell_warning(command: str) -> str | None:
    for pattern, reason in _DANGEROUS_SHELL:
        if pattern.search(command):
            return f"This command {reason}."
    return None


def _url_warning(url: str) -> str | None:
    stripped = url.strip()
    if stripped.lower().startswith("file:"):
        return "This URL reads a local file (file://)."
    host = re.sub(r"^[a-z]+://", "", stripped, flags=re.I)
    if _LOCAL_HOST.match(host):
        return "This URL targets a local/internal address."
    return None


def risk_warning(name: str, args: Mapping[str, object]) -> str | None:
    """A one-line caution for a risky tool call, or ``None`` when nothing stands out.

    Flags destructive/privileged shell commands (``execute``) and fetches aimed at local or
    ``file://`` targets; everything else returns ``None``.
    """
    if name == "execute":
        command = args.get("command")
        if isinstance(command, str):
            return _shell_warning(command)
    elif name == FETCH_TOOL_NAME:
        url = args.get("url")
        if isinstance(url, str):
            return _url_warning(url)
    return None
