"""Path-scoped write gating for ``assembly live --files``.

The ``--files`` brain confirms writes before they touch disk. Rather than gate *every* write
(all-or-nothing, a keypress per write — friction for a hands-free voice turn), a write inside an
``--auto-write`` subtree runs ungated while any other write still pauses for approval — the
allow/interrupt permission model, scoped by path.

This is wired by handing ``HumanInTheLoopMiddleware`` an :class:`InterruptOnConfig` whose ``when``
predicate fires only for writes *outside* every ``--auto-write`` root, so the file-write tools
pause exactly when the target isn't pre-approved. The matching is a transparent posix path-prefix
check (``--auto-write DIR`` is inherently a directory subtree); reads stay ungated and are never
gated here, so no glob/bulk-tool machinery is needed. ``execute`` can't be path-scoped — a command
isn't a single file path — so it stays unconditionally gated: every run is still approved.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest

# The file-mutating tools gated behind human approval when --files is on (reads — incl. grep —
# stay ungated). These take an exact file path, so they are path-scopable: a write under an
# --auto-write subtree skips the gate; a write elsewhere still pauses. execute is gated separately
# (always), since a command isn't a single file path to scope.
_FILE_WRITE_TOOLS = ("write_file", "edit_file")

# Offered to the approver on a gated write, matching deepagents' default for a HITL-gated tool.
# Annotated with the Literal decision types so it satisfies InterruptOnConfig.allowed_decisions.
_ALLOWED_DECISIONS: list[Literal["approve", "edit", "reject", "respond"]] = [
    "approve",
    "edit",
    "reject",
    "respond",
]


def _normalize_target(raw: str) -> tuple[str, ...] | None:
    """The path's ``/``-rooted virtual segments, or ``None`` if it can't be safely localized.

    The model addresses files at ``/``-rooted virtual paths under cwd; a relative path is rooted
    the same way deepagents would. A ``..`` segment can't be statically placed (and the backend
    blocks the traversal anyway), so it returns ``None`` — the caller then fails safe to gating.
    """
    posix = raw.replace("\\", "/")
    if not posix.startswith("/"):
        posix = "/" + posix
    parts = PurePosixPath(posix).parts
    if ".." in parts:
        return None
    return parts


def _under_auto_write(target: tuple[str, ...], roots: Sequence[str]) -> bool:
    """Whether ``target`` (normalized segments) sits in or under one of the ``--auto-write`` roots.

    Compares whole path segments (not a string prefix) so ``/scratchpad`` is *not* treated as
    under the root ``/scratch`` — a string-prefix check would wrongly auto-approve it.
    """
    for root in roots:
        root_parts = PurePosixPath(root).parts
        if target[: len(root_parts)] == root_parts:
            return True
    return False


def _auto_write_gate(auto_write_paths: Sequence[str]) -> Callable[[ToolCallRequest], bool]:
    """A ``when`` predicate: fire the approval interrupt unless the write lands in an auto-write
    subtree. With no auto-write paths nothing is under one, so every write fires — the prior
    all-or-nothing behavior. A non-string or unlocatable (``..``) path fails safe to firing."""

    def when(request: ToolCallRequest) -> bool:
        raw = request.tool_call.get("args", {}).get("file_path")
        if not isinstance(raw, str):
            return True
        target = _normalize_target(raw)
        if target is None:
            return True
        return not _under_auto_write(target, auto_write_paths)

    return when


def write_interrupt_on(auto_write_paths: Sequence[str]) -> dict[str, object]:
    """The ``interrupt_on`` map gating writes for --files, path-scoped by ``--auto-write``.

    ``write_file``/``edit_file`` share one path-scoped :class:`InterruptOnConfig` (pause only
    outside the auto-write subtrees); ``execute`` stays a plain ``True`` so every command run is
    approved. The map is handed to ``create_deep_agent(interrupt_on=…)`` and to the subagent spec.
    """
    from langchain.agents.middleware import InterruptOnConfig

    gate = InterruptOnConfig(
        allowed_decisions=_ALLOWED_DECISIONS, when=_auto_write_gate(auto_write_paths)
    )
    interrupt_on: dict[str, object] = dict.fromkeys(_FILE_WRITE_TOOLS, gate)
    interrupt_on["execute"] = True
    return interrupt_on
