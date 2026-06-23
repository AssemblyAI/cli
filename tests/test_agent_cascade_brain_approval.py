"""Write-approval tests for the deepagents reply brain behind `assembly live`.

Split out of `test_agent_cascade_brain.py` to keep each file under the 500-line gate.
These drive `build_streamer`'s `--files` write-gating against a *real* deepagents graph
wired to a fake chat model (pytest-socket stays armed) — no sockets.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aai_cli.agent_cascade import brain
from aai_cli.agent_cascade.config import CascadeConfig


class FakeChatModel(BaseChatModel):
    """A chat model that replays a scripted list of AIMessages (mirrors the code agent's)."""

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-live-model"

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _gated_graph(model: BaseChatModel, root: str):
    """A real deepagents graph that gates write_file/edit_file, rooted at ``root``."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import InMemorySaver

    return create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        interrupt_on={"write_file": True, "edit_file": True},
        checkpointer=InMemorySaver(),
        system_prompt="be a friendly live agent",
    )


def _write_then(reply: str) -> FakeChatModel:
    """A model that calls write_file once, then (after resume) answers with ``reply``."""
    call = AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": "/n.txt", "content": "hi"}, "id": "w1"}
        ],
    )
    return FakeChatModel(responses=[call, AIMessage(content=reply)])


def test_streamer_approves_write_then_resumes(tmp_path):
    asked: list[tuple[str, dict]] = []

    def approve(name, args):
        asked.append((name, args))
        return True

    graph = _gated_graph(_write_then("Saved your note."), str(tmp_path))
    streamer = brain.build_streamer("k", CascadeConfig(files=True), graph=graph, approver=approve)
    events = list(streamer([{"role": "user", "content": "save a note"}]))
    spoken = "".join(e.text for e in events if isinstance(e, brain.SpeechDelta))
    assert spoken == "Saved your note."
    # The approver was consulted for the write, and the approved write hit the rooted dir.
    assert asked and asked[0][0] == "write_file"
    assert (tmp_path / "n.txt").read_text() == "hi"


def test_streamer_rejects_write_without_approval(tmp_path):
    graph = _gated_graph(_write_then("Okay, I won't save it."), str(tmp_path))
    streamer = brain.build_streamer(
        "k", CascadeConfig(files=True), graph=graph, approver=lambda name, args: False
    )
    events = list(streamer([{"role": "user", "content": "save a note"}]))
    spoken = "".join(e.text for e in events if isinstance(e, brain.SpeechDelta))
    assert spoken == "Okay, I won't save it."
    # Declined: nothing was written to the rooted directory.
    assert not (tmp_path / "n.txt").exists()


def test_streamer_brackets_write_approval_with_pause_events(tmp_path):
    # The human-think wait is bracketed by ApprovalPause(active=True/False) so the engine can
    # suspend its reply-timeout deadline for exactly that interval. The approver runs between
    # the two markers by construction (the streamer yields True, asks, then yields False).
    asked: list[str] = []
    graph = _gated_graph(_write_then("Done."), str(tmp_path))
    streamer = brain.build_streamer(
        "k",
        CascadeConfig(files=True),
        graph=graph,
        approver=lambda name, args: asked.append(name) or True,
    )
    events = list(streamer([{"role": "user", "content": "save"}]))
    pauses = [event.active for event in events if isinstance(event, brain.ApprovalPause)]
    assert pauses == [True, False]  # the write was bracketed: pause on, then resume
    assert asked == ["write_file"]  # the approver was consulted exactly once, for the write
