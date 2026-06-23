"""Tests for the general-purpose subagent spec (`assembly live --files` task tool)."""

from __future__ import annotations

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig
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


def test_graph_kwargs_wires_one_gated_gateway_bound_subagent(monkeypatch, tmp_path):
    # --files binds exactly one subagent: gateway-bound (no model) with every mutating tool gated.
    monkeypatch.chdir(tmp_path)
    subs = brain._graph_kwargs(CascadeConfig(files=True))["subagents"]
    assert isinstance(subs, list) and len(subs) == 1
    spec = subs[0]
    assert spec["name"] == "general-purpose"
    assert "model" not in spec  # inherits the gateway-bound model
    assert spec["interrupt_on"] == {"write_file": True, "edit_file": True, "execute": True}


def test_graph_kwargs_off_binds_no_subagents():
    assert "subagents" not in brain._graph_kwargs(CascadeConfig(files=False))


def test_tool_label_task_is_working_on_a_subtask():
    assert brain._tool_label("task") == "Working on a subtask"
