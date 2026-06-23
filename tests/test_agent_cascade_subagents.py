"""Tests for the live agent's general-purpose subagent (`assembly live` task tool).

deepagents auto-adds a `general-purpose` subagent (its `task` tool) and derives that subagent's
`interrupt_on` from the top-level `create_deep_agent(interrupt_on=…)`, so we don't declare the
subagent ourselves. We only register a harness profile that overrides its prose for a voice turn
(`brain._register_gp_subagent_profile`). These tests guard both halves: the profile override and
the inherited write-gating (a delegated write must still surface at the *parent* approval gate).
"""

from __future__ import annotations

import deepagents
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from aai_cli.agent_cascade import brain, subagents
from aai_cli.agent_cascade.config import CascadeConfig
from tests._cascade_fakes import FakeChatModel


def test_register_gp_subagent_profile_overrides_prompt_and_description(monkeypatch):
    # Capture the registration instead of reading the process-global registry (other tests
    # populate it, and `register_harness_profile` is imported inside the helper, so patching
    # the module attribute is picked up at call time).
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        deepagents,
        "register_harness_profile",
        lambda key, profile: captured.update(key=key, profile=profile),
    )

    subagents.register_gp_subagent_profile()

    # Keyed by the bare provider so the override still applies when --model overrides the
    # default identifier (the gateway model is a ChatOpenAI subclass → provider "openai").
    assert captured["key"] == "openai"
    gp = captured["profile"].general_purpose_subagent
    # Spoken-length summary, not deepagents' "complete answer" default.
    assert "summary" in gp.system_prompt and "transcript" in gp.system_prompt
    assert "subtask" in gp.description and "summary" in gp.description
    # Don't pin a model/tools: the subagent must inherit the gateway-bound model + sandboxed tools.
    assert gp.enabled is None


def test_build_graph_registers_the_gp_subagent_profile(monkeypatch):
    # The override only takes effect if build_graph actually registers it before create_deep_agent.
    keys: list[str] = []
    monkeypatch.setattr(
        deepagents, "register_harness_profile", lambda key, profile: keys.append(key)
    )

    brain.build_graph("k", CascadeConfig(files=False), tools=[], mcp_tools=[])

    assert keys == ["openai"]


def test_profile_override_lands_on_the_auto_added_subagent(monkeypatch, tmp_path):
    # End-to-end: the registered profile must reach deepagents' auto-added general-purpose
    # subagent — proving "openai" matches the gateway model's resolved provider. The task tool's
    # description embeds each subagent's description, so we read it back from the compiled graph.
    monkeypatch.chdir(tmp_path)
    graph = brain.build_graph("k", CascadeConfig(files=True), tools=[], mcp_tools=[])

    task_tool = graph.nodes["tools"].bound.tools_by_name["task"]
    assert subagents._GP_DESCRIPTION in task_tool.description


def test_graph_kwargs_on_gates_writes_without_declaring_a_subagent():
    # --files binds the gating + checkpointer but no explicit subagent: the gateway-bound GP
    # subagent is auto-added and inherits this interrupt_on (see the surfacing test below). The
    # write tools are now path-scoped InterruptOnConfig maps (a `when` predicate), execute a plain
    # True — the auto-added subagent inherits the whole map, so it honors --auto-write too.
    kw = brain._graph_kwargs(CascadeConfig(files=True))
    assert "subagents" not in kw
    interrupt_on = kw["interrupt_on"]
    assert sorted(interrupt_on) == ["edit_file", "execute", "write_file"]
    assert interrupt_on["execute"] is True
    assert "when" in interrupt_on["write_file"] and "when" in interrupt_on["edit_file"]
    assert "checkpointer" in kw and "backend" in kw


def test_graph_kwargs_off_is_empty():
    assert brain._graph_kwargs(CascadeConfig(files=False)) == {}


def test_tool_label_task_is_working_on_a_subtask():
    assert brain._tool_label("task") == "Working on a subtask"


def _delegating_graph(model: BaseChatModel, root: str):
    """A real deepagents graph that gates writes but declares NO subagent — exercising the
    auto-added general-purpose subagent and its inherited top-level interrupt_on (mirrors the
    gated write graph). Inline literals get bidirectional typing; no return annotation so
    pyright accepts it as build_streamer's graph (same shape as the gated-write tests)."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import InMemorySaver

    return create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        interrupt_on={"write_file": True, "edit_file": True},
        checkpointer=InMemorySaver(),
        system_prompt="be a live agent",
    )


def _delegate_then_write(reply: str) -> FakeChatModel:
    """Scripts main -> task(general-purpose) -> subagent -> write_file -> (resume) replies."""
    task_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"description": "save a note", "subagent_type": "general-purpose"},
                "id": "t1",
            }
        ],
    )
    write_call = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/n.txt", "content": "hi"}, "id": "w1"}
        ],
    )
    return FakeChatModel(
        responses=[
            task_call,
            write_call,
            AIMessage(content="subtask done"),
            AIMessage(content=reply),
        ]
    )


def test_subagent_write_surfaces_through_the_parent_gate_and_is_approved(tmp_path):
    # The DECISIVE M2 invariant (the resolved spike): a write delegated to the AUTO-ADDED
    # general-purpose subagent pauses through OUR parent approval loop (build_streamer ->
    # _stream_gated -> _pending_writes -> approver) purely via the inherited top-level
    # interrupt_on. Approved, it lands.
    asked: list[tuple[str, dict]] = []
    graph = _delegating_graph(_delegate_then_write("Saved it via the helper."), str(tmp_path))
    streamer = brain.build_streamer(
        "k",
        CascadeConfig(files=True),
        graph=graph,
        approver=lambda name, args: asked.append((name, args)) or True,
    )

    list(streamer([{"role": "user", "content": "have the helper save a note"}]))

    assert any(name == "write_file" for name, _ in asked)  # the SUBAGENT's write was gated by us
    assert (tmp_path / "n.txt").read_text() == "hi"  # approved -> actually written


def test_subagent_write_is_declined_when_the_approver_rejects(tmp_path):
    graph = _delegating_graph(_delegate_then_write("Okay, left it alone."), str(tmp_path))
    streamer = brain.build_streamer(
        "k", CascadeConfig(files=True), graph=graph, approver=lambda name, args: False
    )

    list(streamer([{"role": "user", "content": "have the helper save a note"}]))

    assert not (tmp_path / "n.txt").exists()  # declined -> nothing written by the subagent
