"""The general-purpose subagent for ``assembly live --files`` (deepagents' ``task`` tool).

One subagent the live agent delegates a focused multi-step subtask to. It OMITS ``model`` (so it
inherits the AssemblyAI gateway-bound model — never a ``provider:model`` string) and ``tools`` (so
it inherits the main sandboxed toolset, keeping its ``execute`` OS-confined). Its ``interrupt_on``
mirrors the main agent's write tools, so the subagent's own mutations prompt through the same
approval loop (verified to surface at the parent gate — see the HITL regression test).
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a focused coworker handling one delegated subtask in the user's project. Work in the "
    "current directory, use the available tools to research or make a contained change, and return "
    "a concise, spoken-length summary of what you did or found — not a transcript."
)


def general_purpose_subagent(interrupt_on: dict[str, bool]) -> dict[str, object]:
    """The ``task`` subagent spec: gateway-bound (no ``model``), full sandboxed tools (no ``tools``),
    with ``interrupt_on`` mirroring the caller's write tools so its mutations stay gated.

    ``interrupt_on`` is a parameter (not a local constant) so this module needn't import
    ``brain._WRITE_TOOLS`` — that would be a circular import, since ``brain`` imports this.
    """
    return {
        "name": "general-purpose",
        "description": (
            "Delegate a focused multi-step subtask — research, gather context, or implement a "
            "contained change — and get back a short summary. Keeps the main voice turn lean."
        ),
        "system_prompt": _SYSTEM_PROMPT,
        "interrupt_on": interrupt_on,
    }
