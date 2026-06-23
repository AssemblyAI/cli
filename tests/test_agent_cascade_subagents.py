"""Tests for the general-purpose subagent spec (`assembly live --files` task tool)."""

from __future__ import annotations

from aai_cli.agent_cascade.subagents import general_purpose_subagent


def test_spec_has_required_keys_and_omits_model_and_tools():
    spec = general_purpose_subagent({"write_file": True, "edit_file": True, "execute": True})
    assert spec["name"] == "general-purpose"
    assert isinstance(spec["description"], str) and spec["description"]
    assert isinstance(spec["system_prompt"], str) and spec["system_prompt"]
    # AssemblyAI-only invariant: no provider:model string — must inherit the gateway-bound model.
    assert "model" not in spec
    # Full-tools path: tools omitted so the subagent inherits the sandboxed main toolset.
    assert "tools" not in spec


def test_spec_interrupt_on_is_the_passed_mapping():
    # Mirrors the caller's write tools verbatim, so the subagent's mutations also prompt. Passing
    # a distinct mapping proves it isn't hardcoded (kills a "return a fixed dict" mutant).
    io = {"write_file": True, "edit_file": True, "execute": True}
    assert general_purpose_subagent(io)["interrupt_on"] == io
    assert general_purpose_subagent({"write_file": True})["interrupt_on"] == {"write_file": True}
