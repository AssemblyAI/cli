"""Tune deepagents' auto-added general-purpose subagent for ``assembly live`` (the ``task`` tool).

deepagents auto-adds a ``general-purpose`` subagent and derives its ``interrupt_on`` from the
top-level ``create_deep_agent(interrupt_on=…)``, so we don't declare the subagent ourselves — it
inherits the gateway-bound model, the sandboxed toolset, *and* the write-gating, and a delegated
write still surfaces at the *parent* approval gate (locked by ``tests/test_agent_cascade_subagents.py``).
The only thing we override is its prose: the SDK's default subagent prompt asks for a "complete
answer", but a live voice turn wants a short, spoken-length summary. We set that (and the
description) through a harness profile, keeping this off ``brain.py`` (which sits at the 500-line gate).
"""

from __future__ import annotations

_GP_SYSTEM_PROMPT = (
    "You are a focused coworker handling one delegated subtask in the user's project. Work in the "
    "current directory, use the available tools to research or make a contained change, and return "
    "a concise, spoken-length summary of what you did or found — not a transcript."
)
_GP_DESCRIPTION = (
    "Delegate a focused multi-step subtask — research, gather context, or implement a "
    "contained change — and get back a short summary. Keeps the main voice turn lean."
)

# The harness-profile registry is keyed by model provider/identifier; the gateway model is a
# ChatOpenAI subclass, so its provider is "openai". We register under the bare provider (not
# provider:model) so the override still applies when --model overrides the default identifier.
# Safe to scope this broadly: brain.build_graph is the *only* create_deep_agent call in the CLI.
_GP_PROFILE_MODEL_PROVIDER = "openai"


def register_gp_subagent_profile() -> None:
    """Override the auto-added general-purpose subagent's prompt + description for a voice turn.

    Registers a harness profile that swaps in a spoken-length summary prompt (instead of
    deepagents' "complete answer" default) and our short description. The subagent keeps
    inheriting the gateway-bound model, the sandboxed toolset, and the top-level ``interrupt_on``.
    Idempotent — re-registers the same profile under the same key; ``brain.build_graph`` calls it
    once per graph build (the deepagents import stays lazy here, off the startup path).
    """
    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        register_harness_profile,
    )

    register_harness_profile(
        _GP_PROFILE_MODEL_PROVIDER,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(
                system_prompt=_GP_SYSTEM_PROMPT, description=_GP_DESCRIPTION
            )
        ),
    )
