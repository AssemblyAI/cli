"""End-to-end tests for the `assembly code` coding agent.

A fake chat model drives the *real* deepagents graph offline (pytest-socket stays
armed), so the filesystem/shell tools, approval interrupt/resume, event rendering, and
REPL loop are all exercised without a network or a TTY.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aai_cli.code_agent import (
    ask_tool,
    cli_tool,
    docs_mcp,
    events,
    fetch_tool,
    firecrawl_search,
    memory,
    skills,
    store,
)
from aai_cli.code_agent.agent import MUTATING_TOOLS, build_agent
from aai_cli.code_agent.events import AssistantText, ErrorText, ToolCall, ToolResult
from aai_cli.code_agent.prompt import build_system_prompt
from aai_cli.code_agent.render import RichRenderer, make_approver
from aai_cli.code_agent.session import QUIT_COMMANDS, CodeSession, run_repl


class FakeChatModel(BaseChatModel):
    """A tool-calling chat model that replays a scripted list of AIMessages."""

    responses: list[AIMessage]
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-code-model"

    def bind_tools(self, tools, **kwargs):
        del tools, kwargs
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        del messages, stop, run_manager, kwargs
        message = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def _write_call(path: str, content: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "write_file", "args": {"file_path": path, "content": content}, "id": "c1"}
        ],
    )


def _session(
    model: BaseChatModel, work: Path, *, approver, auto_approve=False
) -> tuple[CodeSession, list[object]]:
    sink_events: list[object] = []
    agent = build_agent(model=model, root_dir=work, auto_approve=auto_approve)
    session = CodeSession(
        agent=agent, sink=sink_events.append, approver=approver, auto_approve=auto_approve
    )
    return session, sink_events


def test_approved_write_creates_file_and_emits_events(tmp_path: Path) -> None:
    model = FakeChatModel(
        responses=[_write_call("hello.txt", "hi there"), AIMessage(content="Done.")]
    )
    session, sink = _session(model, tmp_path, approver=lambda name, args: True)

    session.send("create hello.txt")

    assert (tmp_path / "hello.txt").read_text() == "hi there"
    assert any(isinstance(e, ToolResult) for e in sink)
    assert any(isinstance(e, AssistantText) and "Done." in e.text for e in sink)


def test_rejected_write_does_not_create_file(tmp_path: Path) -> None:
    model = FakeChatModel(responses=[_write_call("no.txt", "x"), AIMessage(content="Skipped.")])
    seen: list[str] = []

    def reject(name: str, args: dict[str, object]) -> bool:
        seen.append(name)
        return False

    session, _ = _session(model, tmp_path, approver=reject)

    session.send("create no.txt")

    assert not (tmp_path / "no.txt").exists()
    assert seen == ["write_file"]  # the approver was consulted for the gated tool


def test_auto_approve_runs_without_approver_and_announces_calls(tmp_path: Path) -> None:
    model = FakeChatModel(responses=[_write_call("auto.txt", "data"), AIMessage(content="ok")])

    def deny(name, args):  # the approver must never be called under --auto
        raise AssertionError("approver called under auto_approve")

    session, sink = _session(model, tmp_path, approver=deny, auto_approve=True)
    session.send("go")

    assert (tmp_path / "auto.txt").read_text() == "data"
    assert any(isinstance(e, ToolCall) and e.name == "write_file" for e in sink)


def test_run_repl_sends_initial_then_lines_until_quit(tmp_path: Path) -> None:
    model = FakeChatModel(responses=[AIMessage(content="a"), AIMessage(content="b")])
    session, sink = _session(model, tmp_path, approver=lambda name, args: True)
    lines = iter(["", "second", "/quit", "never"])
    run_repl(session, read_line=lambda: next(lines), initial="first")

    texts = [e.text for e in sink if isinstance(e, AssistantText)]
    assert texts == ["a", "b"]  # initial + "second"; blank skipped, stops at /quit


def test_system_prompt_steers_concise_speech() -> None:
    prompt = build_system_prompt("/work")
    assert "/work" in prompt  # anchored to the working directory
    # The prose is read aloud, so the prompt must steer the model to concise, speech-ready
    # replies with code kept out of the spoken text.
    assert "read aloud" in prompt
    assert "fenced code blocks" in prompt
    lowered = prompt.lower()
    assert "concise" in lowered and "spoken" in lowered


def test_mutating_tools_include_cli_shell_and_fetch() -> None:
    assert set(MUTATING_TOOLS) == {"write_file", "edit_file", "execute", "assembly", "fetch_url"}
    assert "exit" in QUIT_COMMANDS and "/exit" in QUIT_COMMANDS


def test_fetch_tool_invokes_fetcher() -> None:
    tool = fetch_tool.build_fetch_tool(lambda url: f"body of {url}")
    assert tool.name == "fetch_url"
    assert tool.invoke({"url": "https://x.test"}) == "body of https://x.test"


def test_ask_tool_uses_bridge_handler() -> None:
    bridge = ask_tool.AskBridge()
    assert "no user" in bridge.ask("q?").lower()  # default before a front-end attaches
    bridge.handler = lambda question: f"answer to {question}"
    tool = ask_tool.build_ask_tool(bridge)
    assert tool.invoke({"question": "deploy now?"}) == "answer to deploy now?"


def test_memory_middleware_creates_dir(tmp_path: Path) -> None:
    root = tmp_path / "mem"
    middleware = memory.build_memory_middleware(root)
    assert root.is_dir()
    assert middleware is not None


def test_checkpointer_in_memory_vs_sqlite(tmp_path, monkeypatch):  # untyped: touches saver.conn
    from langgraph.checkpoint.memory import InMemorySaver

    assert isinstance(store.build_checkpointer(persist=False), InMemorySaver)

    monkeypatch.setattr(store, "sessions_db_path", lambda: tmp_path / "s.sqlite")
    saver = store.build_checkpointer(persist=True)
    assert not isinstance(saver, InMemorySaver)  # a SQLite-backed saver instead
    # Close the underlying connection so it isn't GC'd mid-suite — an unclosed
    # sqlite3.Connection raises PytestUnraisableExceptionWarning on py3.13/Windows,
    # which `filterwarnings=error` turns into a failure in an unrelated later test.
    saver.conn.close()


def test_new_session_id_is_unique_and_short() -> None:
    a = store.new_session_id()
    b = store.new_session_id()
    assert a != b  # each run gets its own thread id (no silent resume of a shared default)
    assert len(a) == 12 and a.isalnum()  # short hex, readable off the splash to resume later


def test_cli_tool_invokes_runner_with_args() -> None:
    captured: list[list[str]] = []

    def runner(args: list[str]) -> str:
        captured.append(args)
        return "ran"

    tool = cli_tool.build_cli_tool(runner)
    out = tool.invoke({"arguments": ["transcribe", "a.mp3"]})
    assert out == "ran"
    assert captured == [["transcribe", "a.mp3"]]


def test_run_assembly_passes_key_via_env_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    cmd_seen: list[str] = []
    env_seen: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        cmd_seen.extend(cmd)
        env_seen.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr("aai_cli.code_agent.cli_tool.subprocess.run", fake_run)
    result = cli_tool.run_assembly(["transcripts", "list"], api_key="secret-key")

    assert "secret-key" not in " ".join(cmd_seen)  # never on argv
    assert env_seen["ASSEMBLYAI_API_KEY"] == "secret-key"  # passed via env
    assert "exit code: 0" in result and "ok" in result


def test_docs_mcp_load_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(url):
        raise RuntimeError("blocked host")

    # Replace the coroutine factory with a sync raiser so no un-awaited coroutine is
    # created; load_docs_tools must swallow the failure and report no docs tools.
    monkeypatch.setattr(docs_mcp, "_fetch", boom)
    assert docs_mcp.load_docs_tools("https://example.invalid") == []


def test_build_skills_present_and_absent(tmp_path: Path) -> None:
    assert skills.build_skills(tmp_path) is None  # empty dir -> no skills, no tool

    skill_dir = tmp_path / "assemblyai"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: assemblyai\ndescription: x\n---\nbody")
    bundle = skills.build_skills(tmp_path)
    assert bundle is not None  # constructing the middleware also validates the custom prompt
    _middleware, reader = bundle
    assert reader.name == skills.READ_SKILL_TOOL_NAME


def test_read_skill_tool_reads_under_root_and_blocks_escape(tmp_path: Path) -> None:
    skill_dir = tmp_path / "assemblyai"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("the skill body")
    (tmp_path.parent / "secret.md").write_text("top secret")

    reader = skills.build_skill_reader(tmp_path)
    # The path is the prompt's backend-virtual form (leading slash, relative to root).
    assert reader.invoke({"path": "/assemblyai/SKILL.md"}) == "the skill body"
    # A traversal out of the skills dir is refused (not the neighbouring file's contents).
    escaped = reader.invoke({"path": "/../secret.md"})
    assert "outside the skills directory" in escaped and "top secret" not in escaped
    # A missing skill file reports an error rather than raising.
    assert "not found" in reader.invoke({"path": "/assemblyai/MISSING.md"})


def test_web_search_tool_gated_on_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    assert firecrawl_search.build_web_search_tool() is None

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
    tool = firecrawl_search.build_web_search_tool()
    assert tool is not None and tool.name == "firecrawl_search"


def test_message_events_coerces_list_content() -> None:
    msg = AIMessage(content=[{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
    out = events.message_events(msg, announce_calls=False)
    assert out == [AssistantText("foobar")]


def test_rich_renderer_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = RichRenderer()
    renderer(AssistantText("hi"))
    renderer(ToolCall(name="write_file", args={"file_path": "a"}))
    renderer(ToolResult(name="write_file", content="Updated a"))
    approver = make_approver(lambda name, args: True)
    assert approver("write_file", {}) is True
    out = capsys.readouterr().out
    assert "hi" in out and "write_file" in out


# --- slice-unit edge cases (cover the lazy bodies + error/guard branches) -----


def test_fetch_url_fetches_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    class Resp:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(httpx, "get", lambda url, **kw: Resp("body"))
    assert fetch_tool.fetch_url("https://x.test") == "body"

    big = "y" * (fetch_tool._MAX_CHARS + 10)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: Resp(big))
    out = fetch_tool.fetch_url("https://x.test")
    assert out.endswith("…[truncated]") and len(out) < len(big) + 20


def test_load_docs_tools_success(monkeypatch):  # untyped: tools list compares to str sentinels
    class FakeClient:
        def __init__(self, connections):
            self.connections = connections

        async def get_tools(self):
            return ["docs-tool"]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient)
    assert docs_mcp.load_docs_tools("https://docs.test") == ["docs-tool"]


def test_config_root_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/cfg")
    assert memory.memory_root() == Path("/tmp/cfg/code-memory")
    assert skills.skills_root() == Path("/tmp/cfg/skills")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert memory.memory_root() == Path.home() / ".claude" / "code-memory"
    assert skills.skills_root() == Path.home() / ".claude" / "skills"

    monkeypatch.setattr("platformdirs.user_data_dir", lambda app: str(tmp_path))
    db = store.sessions_db_path()
    assert db == tmp_path / "code-sessions" / "sessions.sqlite"
    assert db.parent.is_dir()


def test_event_helpers_fallbacks() -> None:
    assert events._text_of(123) == "123"  # neither str nor list
    assert events.new_messages({}, 0) == []  # no "messages" key
    assert events.interrupt_request({}) is None


def test_session_surfaces_turn_failure_as_error_event() -> None:
    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("gateway 500")

    seen: list[object] = []
    session = CodeSession(agent=Boom(), sink=seen.append, approver=lambda n, a: True)
    session.send("go")
    assert any(isinstance(e, ErrorText) and "gateway 500" in e.text for e in seen)


def test_session_propagates_keyboard_interrupt() -> None:
    class Stop:
        def invoke(self, *a, **k):
            raise KeyboardInterrupt

    session = CodeSession(agent=Stop(), sink=lambda e: None, approver=lambda n, a: True)
    with pytest.raises(KeyboardInterrupt):
        session.send("go")


def test_decide_coerces_non_dict_args() -> None:
    seen: dict[str, object] = {}

    class Dummy:
        def invoke(self, *a, **k):
            return {"messages": []}

    session = CodeSession(
        agent=Dummy(), sink=lambda e: None, approver=lambda n, a: seen.update(a=a) or True
    )
    decision = session._decide({"name": "t", "args": "not-a-dict"})
    assert decision == {"type": "approve"} and seen["a"] == {}


def test_run_repl_stops_on_eof() -> None:
    class Dummy:
        def invoke(self, *a, **k):
            return {"messages": []}

    session = CodeSession(agent=Dummy(), sink=lambda e: None, approver=lambda n, a: True)
    run_repl(session, read_line=lambda: None)  # immediate EOF -> returns without error


def test_rich_renderer_renders_error(capsys: pytest.CaptureFixture[str]) -> None:
    RichRenderer()(ErrorText("boom happened"))
    assert "boom happened" in capsys.readouterr().err


def test_cli_tool_truncates_and_includes_stderr() -> None:
    import subprocess

    long = "z" * (cli_tool._MAX_OUTPUT_CHARS + 50)
    assert cli_tool._truncate(long).endswith("…[output truncated]")
    proc = subprocess.CompletedProcess(["x"], 1, stdout="out", stderr="boom")
    rendered = cli_tool._format_result(proc)
    assert "exit code: 1" in rendered and "stderr:\nboom" in rendered


def test_rich_renderer_notice(capsys: pytest.CaptureFixture[str]) -> None:
    RichRenderer().notice("heads up")
    assert "heads up" in capsys.readouterr().err


def test_rich_renderer_escapes_markup(capsys: pytest.CaptureFixture[str]) -> None:
    renderer = RichRenderer()
    renderer(AssistantText("[bold]x[/bold]"))
    renderer(ToolCall(name="t", args={"a": "[red]"}))
    renderer(ToolResult(name="t", content="[u]z[/u]"))
    renderer(ErrorText("[i]e[/i]"))
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # Without escaping, Rich would consume these as style tags (and strip the brackets);
    # escaped, the literal brackets survive in the output.
    assert "[bold]" in combined and "[red]" in combined
    assert "[u]" in combined and "[i]" in combined
