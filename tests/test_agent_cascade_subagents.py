"""Tests for the general-purpose subagent spec (`assembly live --files` task tool)."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from aai_cli.agent_cascade import brain, streamer
from aai_cli.agent_cascade.config import CascadeConfig
from aai_cli.agent_cascade.subagents import general_purpose_subagent
from tests._cascade_fakes import FakeChatModel


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


def _delegating_graph(model: BaseChatModel, root: str):
    """A real deepagents graph that binds a gated general-purpose subagent (mirrors the gated
    write graph). Inline literals get bidirectional typing; no return annotation so pyright
    accepts it as build_streamer's graph (same shape as the gated-write tests)."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.subagents import SubAgent
    from langgraph.checkpoint.memory import InMemorySaver

    spec: SubAgent = {
        "name": "general-purpose",
        "description": "delegate a focused subtask and return a summary",
        "system_prompt": "be a focused helper; return a concise summary",
        "interrupt_on": {"write_file": True, "edit_file": True},
    }
    return create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        interrupt_on={"write_file": True, "edit_file": True},
        checkpointer=InMemorySaver(),
        subagents=[spec],
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
    # The DECISIVE M2 invariant (the resolved spike): a subagent's write pauses through OUR parent
    # approval loop (build_streamer -> _stream_gated -> _pending_writes -> approver). Approved, it lands.
    asked: list[tuple[str, dict]] = []
    graph = _delegating_graph(_delegate_then_write("Saved it via the helper."), str(tmp_path))
    stream_reply = streamer.build_streamer(
        "k",
        CascadeConfig(files=True),
        graph=graph,
        approver=lambda name, args: asked.append((name, args)) or True,
    )

    list(stream_reply([{"role": "user", "content": "have the helper save a note"}]))

    assert any(name == "write_file" for name, _ in asked)  # the SUBAGENT's write was gated by us
    assert (tmp_path / "n.txt").read_text() == "hi"  # approved -> actually written


def test_subagent_write_is_declined_when_the_approver_rejects(tmp_path):
    graph = _delegating_graph(_delegate_then_write("Okay, left it alone."), str(tmp_path))
    stream_reply = streamer.build_streamer(
        "k", CascadeConfig(files=True), graph=graph, approver=lambda name, args: False
    )

    list(stream_reply([{"role": "user", "content": "have the helper save a note"}]))

    assert not (tmp_path / "n.txt").exists()  # declined -> nothing written by the subagent
