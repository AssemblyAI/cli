# `assembly live` File Read/Write Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `assembly live` read, write, and search files in its launch directory, opt-in behind `--files`, with writes confirmed by a `y/n` keypress in the voice TUI.

**Architecture:** `assembly live`'s deepagents brain already binds the filesystem toolset (`read_file`/`write_file`/`edit_file`/`ls`/`glob`/`grep`) — but against an in-memory backend, so today it touches ephemeral graph state, not disk, and is unadvertised. The flag flips three switches in `aai_cli/agent_cascade/brain.py::build_graph`: (1) point the backend at the real cwd via `FilesystemBackend(root_dir=cwd, virtual_mode=True)`; (2) gate `write_file`/`edit_file` with `interrupt_on` + an `InMemorySaver` checkpointer; (3) advertise the capability in the system prompt. The brain's completer resolves write interrupts through an injected `Approver` (the exact pattern `aai_cli/code_agent/session.py` uses); the voice TUI supplies it by reusing `code_agent.modals.ApprovalScreen`, and headless runs auto-deny.

**Tech Stack:** Python 3.12+, deepagents / langgraph / langchain, Typer, Textual, pytest + syrupy, `uv`.

## Global Constraints

- **Opt-in.** New boolean flag `--files`, default off. With it off, behavior is byte-for-byte unchanged (default in-memory backend, no gating, nothing advertised).
- **Reads ungated, incl. `grep`.** `read_file`/`ls`/`glob`/`grep` auto-approve. Only `write_file`/`edit_file` are gated.
- **Writes confirmed via TUI keypress** (`y`/`a`/`n`). Headless / non-TTY runs **auto-deny** writes.
- **Rooted at cwd**, `FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)` — traversal escapes blocked.
- **No shell.** `execute` stays bound but inert (no sandbox backend); it is **not** advertised and **not** gated.
- **Reply timeout excludes human-approval wait** — with `--files` on, the reply leg runs without the 60s wall-clock backstop (a keypress may take arbitrarily long).
- **Repo gates:** TDD; `from __future__ import annotations` atop every module; modern typing (`X | None`); errors→stderr/data→stdout; help copy terse, imperative, **no trailing period**; `--help`/TUI snapshots regenerated with `--snapshot-update`, never hand-edited. The CI gate (`./scripts/check.sh`) enforces 100% patch coverage **and** a diff-scoped mutation gate — assert behavior that would *fail* if a changed line broke, not just execute it. Run the full gate to green before the final commit.
- **Commit discipline:** the pre-commit hook blocks `git commit` unless `./scripts/check.sh` last passed for the current tree. Per-task commits during iteration may use `AAI_ALLOW_COMMIT=1 git commit …`; the **final** task runs the full gate and commits without the override.

---

### Task 1: Real-cwd backend + write-gating in `build_graph`

Add the `files` config knob and make `build_graph` swap to a real-cwd `FilesystemBackend` with write-gating + checkpointer when it's set. Isolate the gating decision in a pure, directly-testable helper (`_graph_kwargs`) so we never introspect langgraph internals.

**Files:**
- Modify: `aai_cli/agent_cascade/config.py` (add `files` field)
- Modify: `aai_cli/agent_cascade/brain.py` (`_WRITE_TOOLS`, `_build_fs_backend`, `_graph_kwargs`, wire into `build_graph`)
- Test: `tests/test_agent_cascade_config.py`, `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Produces: `CascadeConfig.files: bool` (default `False`); `brain._WRITE_TOOLS: tuple[str, ...] = ("write_file", "edit_file")`; `brain._build_fs_backend() -> BackendProtocol`; `brain._graph_kwargs(config: CascadeConfig, *, backend_factory: Callable[[], object] = _build_fs_backend) -> dict[str, object]` returning `{}` when `not config.files` and `{"backend", "interrupt_on", "checkpointer"}` when set.

- [ ] **Step 1: Write the failing config test**

In `tests/test_agent_cascade_config.py`:

```python
def test_files_defaults_off():
    from aai_cli.agent_cascade.config import CascadeConfig

    assert CascadeConfig().files is False
```

- [ ] **Step 2: Run it, verify it fails**

Run: `uv run pytest tests/test_agent_cascade_config.py::test_files_defaults_off -q`
Expected: FAIL — `AttributeError: 'CascadeConfig' object has no attribute 'files'`.

- [ ] **Step 3: Add the config field**

In `aai_cli/agent_cascade/config.py`, inside `CascadeConfig` (after `format_turns`):

```python
    # Opt-in: let the agent read/write files in the launch directory (writes are gated).
    files: bool = False
```

- [ ] **Step 4: Write the failing `_graph_kwargs` tests**

In `tests/test_agent_cascade_brain.py`:

```python
def test_graph_kwargs_empty_when_files_off():
    from aai_cli.agent_cascade import brain
    from aai_cli.agent_cascade.config import CascadeConfig

    assert brain._graph_kwargs(CascadeConfig(files=False)) == {}


def test_graph_kwargs_gates_writes_and_roots_backend_at_cwd(monkeypatch, tmp_path):
    from deepagents.backends import FilesystemBackend

    from aai_cli.agent_cascade import brain
    from aai_cli.agent_cascade.config import CascadeConfig

    monkeypatch.chdir(tmp_path)
    kwargs = brain._graph_kwargs(CascadeConfig(files=True))

    backend = kwargs["backend"]
    assert isinstance(backend, FilesystemBackend)
    # FilesystemBackend resolves the root to `cwd`; virtual_mode blocks traversal escapes.
    from pathlib import Path

    assert Path(backend.cwd) == tmp_path.resolve()
    assert backend.virtual_mode is True
    # Only the mutating file tools are gated; reads (incl. grep) stay ungated.
    assert kwargs["interrupt_on"] == {"write_file": True, "edit_file": True}
    assert "execute" not in kwargs["interrupt_on"]
    assert kwargs["checkpointer"] is not None
```

- [ ] **Step 5: Run them, verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k graph_kwargs -q`
Expected: FAIL — `AttributeError: module 'aai_cli.agent_cascade.brain' has no attribute '_graph_kwargs'`.

- [ ] **Step 6: Implement the helpers and wire `build_graph`**

In `aai_cli/agent_cascade/brain.py`, add the import near the top:

```python
from pathlib import Path
```

Add module-level constant (near `_TOOL_LABELS`):

```python
# The mutating file tools gated behind human approval when --files is on (reads — incl. grep —
# stay ungated). Matches the code agent's write-tool names so the same approval flow applies.
_WRITE_TOOLS = ("write_file", "edit_file")
```

Add the backend factory + kwargs helper (above `build_graph`):

```python
def _build_fs_backend() -> object:
    """A deepagents filesystem backend rooted at the launch directory.

    ``virtual_mode=True`` maps the model's ``/``-rooted paths under cwd and blocks traversal
    escapes — the same containment ``assembly code`` gets from its ``LocalShellBackend``. This
    is a filesystem (not sandbox) backend, so the always-bound ``execute`` tool stays inert.
    """
    from deepagents.backends import FilesystemBackend

    return FilesystemBackend(root_dir=str(Path.cwd()), virtual_mode=True)


def _graph_kwargs(
    config: CascadeConfig, *, backend_factory: Callable[[], object] = _build_fs_backend
) -> dict[str, object]:
    """The extra ``create_deep_agent`` kwargs that turn on real-cwd files + write-gating.

    Empty when ``--files`` is off, so the graph is built exactly as before. When on: a real-cwd
    backend, ``interrupt_on`` pausing only the mutating tools for human approval, and an
    in-memory checkpointer (interrupt/resume needs one). ``backend_factory`` is the test seam.
    """
    if not config.files:
        return {}
    from langgraph.checkpoint.memory import InMemorySaver

    return {
        "backend": backend_factory(),
        "interrupt_on": dict.fromkeys(_WRITE_TOOLS, True),
        "checkpointer": InMemorySaver(),
    }
```

Then in `build_graph`, replace the `return create_deep_agent(...)` call with:

```python
    return create_deep_agent(
        model=model,
        tools=builtin + extra,
        system_prompt=build_system_prompt(config.system_prompt, tools=builtin, extra_tools=extra),
        **_graph_kwargs(config),
    )
```

(`Callable` is already imported in `brain.py`.)

- [ ] **Step 7: Run the tests, verify they pass**

Run: `uv run pytest tests/test_agent_cascade_config.py::test_files_defaults_off tests/test_agent_cascade_brain.py -k graph_kwargs -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add aai_cli/agent_cascade/config.py aai_cli/agent_cascade/brain.py tests/test_agent_cascade_config.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): real-cwd filesystem backend + write-gating behind --files"
```

---

### Task 2: Advertise the file capability + speakable tool labels

Tell the model it can read/write/search files (only when `--files` is on), and give the file tools speakable affordance labels so the live UI shows "Writing a file…" instead of sitting silent.

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (`build_system_prompt` gains `files`; `_TOOL_LABELS` additions)
- Test: `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Consumes: `CascadeConfig.files` (Task 1).
- Produces: `brain.build_system_prompt(persona: str, *, tools, extra_tools=(), files: bool = False) -> str`; expanded `brain._TOOL_LABELS`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_cascade_brain.py`:

```python
def test_system_prompt_advertises_files_when_enabled():
    prompt = brain.build_system_prompt("You are a helper.", tools=[], files=True)
    assert "read" in prompt and "write" in prompt and "files" in prompt
    assert "working directory" in prompt


def test_system_prompt_omits_files_when_disabled():
    prompt = brain.build_system_prompt("You are a helper.", tools=[], files=False)
    # No tools and no files -> the no-tools guidance, which must not claim file access.
    assert "working directory" not in prompt


def test_tool_label_for_write_is_speakable():
    assert brain._tool_label("write_file") == "Writing a file"
    assert brain._tool_label("grep") == "Searching files"
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k "system_prompt_advertises_files or system_prompt_omits_files or tool_label_for_write" -q`
Expected: FAIL — `build_system_prompt() got an unexpected keyword argument 'files'`.

- [ ] **Step 3: Implement**

In `aai_cli/agent_cascade/brain.py`, extend `_TOOL_LABELS`:

```python
_TOOL_LABELS = {
    WEB_SEARCH_TOOL_NAME: "Searching the web",
    "read_file": "Reading a file",
    "write_file": "Writing a file",
    "edit_file": "Editing a file",
    "ls": "Listing files",
    "glob": "Finding files",
    "grep": "Searching files",
}
```

Add the capability phrase constant (near `_NO_TOOLS_GUIDANCE`):

```python
# Advertised when --files is on, so the model knows it can touch the launch directory (and the
# spoken tail still keeps replies short). Writes pause for the user's y/n; reads are immediate.
_FILE_CAPABILITY = "read, write, and search files in your working directory"
```

Change `build_system_prompt` to accept `files` and fold the phrase into the capability clause:

```python
def build_system_prompt(
    persona: str,
    *,
    tools: Sequence[BaseTool],
    extra_tools: Sequence[BaseTool] = (),
    files: bool = False,
) -> str:
    capabilities = _tool_capabilities(tools)
    extra = _extra_capability(extra_tools)
    if extra is not None:
        capabilities.append(extra)
    if files:
        capabilities.append(_FILE_CAPABILITY)
    if not capabilities:
        return f"{persona}\n\n{_NO_TOOLS_GUIDANCE}"
    guidance = (
        f"You can use tools to help answer: {_join_clause(capabilities)}. Reach for a "
        "tool when a question needs fresh or external information; answer directly and "
        "instantly when you already know. Only offer to do what these tools allow — don't "
        f"say you'll search the web or look something up unless it's listed here. {_SPOKEN_TAIL}"
    )
    return f"{persona}\n\n{guidance}"
```

Update the `build_graph` call from Task 1 to pass `files=config.files`:

```python
        system_prompt=build_system_prompt(
            config.system_prompt, tools=builtin, extra_tools=extra, files=config.files
        ),
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k "system_prompt or tool_label" -q`
Expected: PASS (existing prompt tests still pass — `files` defaults to `False`).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): advertise file capability + speakable file tool labels"
```

---

### Task 3: Resolve write-approval interrupts in the completer

When the gated graph pauses on a write, ask an injected `Approver` and resume with approve/reject — looping until the turn finishes. Reuse `code_agent.events.interrupt_request`. Use a fresh per-turn `thread_id` so the checkpointer never accumulates state across the cascade's full-history-per-turn calls.

**Files:**
- Modify: `aai_cli/agent_cascade/brain.py` (`Approver`, `build_completer` gains `approver`; `_run_graph`/`_drive_graph` gain `config`; new `_resolve_writes`/`_decide`)
- Test: `tests/test_agent_cascade_brain.py`

**Interfaces:**
- Consumes: `brain._WRITE_TOOLS`, `CascadeConfig.files` (Task 1); `aai_cli.code_agent.events.interrupt_request`.
- Produces: `brain.Approver = Callable[[str, dict[str, object]], bool]`; `brain.build_completer(api_key, config, *, graph=None, approver: Approver | None = None) -> Callable[..., str]`.

- [ ] **Step 1: Write the failing approval tests**

In `tests/test_agent_cascade_brain.py` (the `FakeChatModel` + `create_deep_agent` helpers already exist in this file). Add a real gated graph builder and two tests:

```python
def _gated_graph(model: BaseChatModel):
    """A real deepagents graph that gates write_file (mirrors --files), for approval tests."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend
    from langgraph.checkpoint.memory import InMemorySaver

    return create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir="/", virtual_mode=True),
        interrupt_on={"write_file": True, "edit_file": True},
        checkpointer=InMemorySaver(),
        system_prompt="be a friendly live agent",
    )


def _write_then_done():
    """A model that first calls write_file, then (after resume) answers in plain text."""
    call = AIMessage(
        content="",
        tool_calls=[{"name": "write_file", "args": {"file_path": "/notes.txt", "content": "hi"}, "id": "w1"}],
    )
    return FakeChatModel(responses=[call, AIMessage(content="Saved your note.")])


def test_write_is_approved_then_resumes(monkeypatch):
    asked: list[tuple[str, dict]] = []

    def approve(name, args):
        asked.append((name, args))
        return True

    graph = _gated_graph(_write_then_done())
    complete = brain.build_completer("k", CascadeConfig(files=True), graph=graph, approver=approve)
    reply = complete([{"role": "user", "content": "save a note"}])
    assert reply == "Saved your note."
    assert asked and asked[0][0] == "write_file"


def test_write_is_rejected_without_approval():
    graph = _gated_graph(
        FakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {"file_path": "/n.txt", "content": "x"}, "id": "w1"}],
                ),
                AIMessage(content="Okay, I won't save it."),
            ]
        )
    )
    complete = brain.build_completer(
        "k", CascadeConfig(files=True), graph=graph, approver=lambda name, args: False
    )
    reply = complete([{"role": "user", "content": "save a note"}])
    assert reply == "Okay, I won't save it."
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run pytest tests/test_agent_cascade_brain.py -k "write_is_approved or write_is_rejected" -q`
Expected: FAIL — `build_completer() got an unexpected keyword argument 'approver'`.

- [ ] **Step 3: Implement the approver wiring**

In `aai_cli/agent_cascade/brain.py`, add the import + `itertools` and type alias near the top:

```python
import itertools
```

Below `_FLOW_LOG`/imports, add:

```python
# Decide whether a gated write may run (front-end supplied). Mirrors the code agent's Approver.
Approver = Callable[[str, dict[str, object]], bool]

# Message handed back to the model when the user declines a write (matches the code agent's copy).
_DECLINED = "User declined to run this tool."
```

Rewrite `build_completer` to thread the approver and a fresh per-turn config:

```python
def build_completer(
    api_key: str,
    config: CascadeConfig,
    *,
    graph: CompiledAgent | None = None,
    approver: Approver | None = None,
) -> Callable[..., str]:
    """A ``complete_reply`` for the cascade engine backed by the deepagents graph.

    When ``--files`` gates writes, the graph pauses on a write; ``approver`` decides and the
    turn resumes (see :func:`_resolve_writes`). Each turn uses a fresh ``thread_id`` so the
    checkpointer never accumulates the cascade's full-history-per-turn input across turns.
    ``graph``/``approver`` are injected in tests.
    """
    resolved = build_graph(api_key, config) if graph is None else graph
    turn_ids = itertools.count()

    def complete_reply(
        messages: list[ChatCompletionMessageParam],
        on_tool: Callable[[str], None] | None = None,
    ) -> str:
        conversation = [message for message in messages if message.get("role") != "system"]
        run_config = (
            {"configurable": {"thread_id": f"live-{next(turn_ids)}"}} if config.files else None
        )
        return _reply_text(_run_graph(resolved, conversation, on_tool, approver, run_config))

    return complete_reply
```

Thread `approver`/`config` through `_run_graph` and `_drive_graph`, and add the resolution loop:

```python
def _run_graph(
    graph: CompiledAgent,
    conversation: list[ChatCompletionMessageParam],
    on_tool: Callable[[str], None] | None = None,
    approver: Approver | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    try:
        result = _drive_graph(graph, {"messages": conversation}, on_tool, config)
        return _resolve_writes(graph, result, approver, on_tool, config)
    except CLIError:
        raise
    except Exception as exc:
        raise CLIError(
            f"the agent couldn't complete the turn: {exc}", error_type="agent_brain_error"
        ) from exc


def _resolve_writes(
    graph: CompiledAgent,
    result: dict[str, object],
    approver: Approver | None,
    on_tool: Callable[[str], None] | None,
    config: dict[str, object] | None,
) -> dict[str, object]:
    """Loop approving/rejecting gated writes until the turn no longer pauses.

    A no-op when nothing is gated (``--files`` off): ``interrupt_request`` returns ``None`` and
    the initial result is returned unchanged.
    """
    from langgraph.types import Command

    from aai_cli.code_agent.events import interrupt_request

    while True:
        request = interrupt_request(result)
        if request is None:
            return result
        actions = request.get("action_requests")
        actions = actions if isinstance(actions, list) else []
        decisions = [_decide(action, approver) for action in actions]
        result = _drive_graph(graph, Command(resume={"decisions": decisions}), on_tool, config)


def _decide(action: dict[str, object], approver: Approver | None) -> dict[str, object]:
    """Ask the approver about one pending write and shape the resume decision (reject if none)."""
    name = str(action.get("name", ""))
    args = action.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if approver is not None and approver(name, args):
        return {"type": "approve"}
    return {"type": "reject", "message": _DECLINED}
```

Update `_drive_graph` to accept and forward `config` (replace its body's `None`):

```python
def _drive_graph(
    graph: CompiledAgent,
    graph_input: object,
    on_tool: Callable[[str], None] | None = None,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    if (on_tool is not None or debuglog.active()) and hasattr(graph, "stream"):
        last: dict[str, object] = {}
        seen = 0
        for chunk in graph.stream(graph_input, config, stream_mode="values"):
            seen = _log_flow(chunk, seen, on_tool)
            last = chunk
        return last
    return graph.invoke(graph_input, config)
```

(Note: `_drive_graph`'s `graph_input` is now `object` — it accepts a `Command` on resume.)

- [ ] **Step 4: Run the tests, verify they pass**

Run: `uv run pytest tests/test_agent_cascade_brain.py -q`
Expected: PASS (all, including the pre-existing completer tests — `approver`/`config` default to `None`, so the non-`files` path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/brain.py tests/test_agent_cascade_brain.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): resolve write-approval interrupts in the reply completer"
```

---

### Task 4: Thread the approver through the engine + drop the timeout when files are on

`CascadeDeps.real` passes the approver to `build_completer`; the reply leg runs without the 60s backstop when `--files` is on (a keypress can pause arbitrarily long).

**Files:**
- Modify: `aai_cli/agent_cascade/engine.py` (`CascadeDeps.real` gains `approver`; `_complete_within` accepts `timeout: float | None`; `_generate_reply` chooses the timeout)
- Test: `tests/test_agent_cascade_engine.py`, `tests/_cascade_fakes.py` (only if a helper needs the new kwarg)

**Interfaces:**
- Consumes: `brain.Approver`, `brain.build_completer(..., approver=...)` (Task 3); `CascadeConfig.files` (Task 1).
- Produces: `CascadeDeps.real(api_key, config, *, audio, stt_params, approver: brain.Approver | None = None)`; `CascadeSession._complete_within(messages, timeout: float | None) -> str` (runs inline when `timeout is None`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_cascade_engine.py`:

```python
def test_complete_within_runs_inline_when_no_timeout():
    # With --files on the reply leg runs with no wall-clock deadline (human approval can pause),
    # so complete_reply runs on the *calling* thread rather than a timeout child thread.
    seen: list[int] = []

    def reply(messages, on_tool=None):
        seen.append(threading.get_ident())
        return "ok"

    session, _r, _p = make_session(complete_reply=reply, config=CascadeConfig(files=True))
    out = session._complete_within([{"role": "user", "content": "hi"}], None)
    assert out == "ok"
    assert seen == [threading.get_ident()]  # ran inline, not on a child thread


def test_complete_within_uses_child_thread_when_timed():
    seen: list[int] = []

    def reply(messages, on_tool=None):
        seen.append(threading.get_ident())
        return "ok"

    session, _r, _p = make_session(complete_reply=reply)
    session._complete_within([{"role": "user", "content": "hi"}], 60.0)
    assert seen and seen[0] != threading.get_ident()  # ran on the timeout child thread


def test_real_passes_approver_to_completer(monkeypatch):
    captured: dict[str, object] = {}

    def fake_build_completer(api_key, config, *, approver=None):
        captured["approver"] = approver
        return lambda messages, on_tool=None: ""

    monkeypatch.setattr(engine.brain, "build_completer", fake_build_completer)
    sentinel = lambda name, args: True
    engine.CascadeDeps.real(
        "k", CascadeConfig(files=True), audio=iter([]), stt_params=object(), approver=sentinel
    )
    assert captured["approver"] is sentinel
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run pytest tests/test_agent_cascade_engine.py -k "complete_within or real_passes_approver" -q`
Expected: FAIL — `_complete_within()` signature mismatch / `real()` has no `approver` kwarg.

- [ ] **Step 3: Implement**

In `aai_cli/agent_cascade/engine.py`, make `_complete_within` accept `float | None` and run inline when `None`:

```python
    def _complete_within(
        self, messages: list[ChatCompletionMessageParam], timeout: float | None
    ) -> str:
        """Run the blocking reply leg, optionally under a wall-clock backstop.

        ``timeout=None`` (used when ``--files`` gates writes) runs ``complete_reply`` inline:
        a write pauses for a human ``y/n`` keypress that may take arbitrarily long, so the
        60s backstop must not fire. Otherwise the leg runs on a throwaway daemon thread and is
        cut off after ``timeout`` so a stuck network leg can't hang the turn forever.
        """
        if timeout is None:
            return self.deps.complete_reply(messages, on_tool=self.renderer.tool_call)
        replies: list[str] = []
        failures: list[CLIError] = []

        def run() -> None:
            try:
                replies.append(self.deps.complete_reply(messages, on_tool=self.renderer.tool_call))
            except CLIError as exc:
                failures.append(exc)

        worker = threading.Thread(target=run, daemon=True)  # pragma: no mutate
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            raise CLIError(
                f"the agent took longer than {timeout:.0f}s to respond and was cut off",
                error_type="agent_timeout",
            )
        if failures:
            raise failures[0]
        return replies[0]
```

In `_generate_reply`, choose the timeout by `files`:

```python
        timeout = None if self.config.files else _REPLY_TIMEOUT_SECONDS
        try:
            reply = self._complete_within(messages, timeout)
```

In `CascadeDeps.real`, add the `approver` parameter and pass it through:

```python
    @classmethod
    def real(
        cls,
        api_key: str,
        config: CascadeConfig,
        *,
        audio: Iterable[bytes],
        stt_params: StreamingParameters,
        approver: brain.Approver | None = None,
    ) -> CascadeDeps:
        def run_stt(on_turn: Callable[[object], None]) -> None:
            client.stream_audio(api_key, audio, params=stt_params, on_turn=on_turn)

        complete_reply = brain.build_completer(api_key, config, approver=approver)

        def synthesize(text: str) -> bytes:
            spec = SpeakConfig(
                text=text,
                voice=config.voice,
                language=config.language,
                sample_rate=TTS_SAMPLE_RATE,
                extra=config.tts_extra,
            )
            return tts_session.synthesize(api_key, spec).pcm

        return cls(run_stt=run_stt, complete_reply=complete_reply, synthesize=synthesize)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `uv run pytest tests/test_agent_cascade_engine.py -q`
Expected: PASS (existing timeout test still green — the timed branch is unchanged).

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/engine.py tests/test_agent_cascade_engine.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): pass write approver through engine; skip reply timeout when files on"
```

---

### Task 5: TUI write-approval modal (reuse `ApprovalScreen`)

Give `LiveAgentApp` an `approve_write` that blocks the cascade worker on the code agent's `ApprovalScreen` (keyboard `y`/`a`/`n`) and returns the decision. `a` (auto) approves all later writes this session.

**Files:**
- Modify: `aai_cli/agent_cascade/tui.py` (`_auto_approve_writes`, `_modal_result`, `approve_write`, transparent-modal CSS)
- Test: `tests/test_live_tui.py`

**Interfaces:**
- Consumes: `aai_cli.code_agent.modals.ApprovalScreen` (a `ModalScreen[str]` returning `"approve"`/`"auto"`/`"reject"`).
- Produces: `LiveAgentApp.approve_write(name: str, args: dict[str, object]) -> bool`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_live_tui.py`:

```python
def test_approve_write_returns_true_on_approve(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_modal_result", lambda screen, default: "approve")
    assert app.approve_write("write_file", {"file_path": "/n.txt"}) is True


def test_approve_write_returns_false_on_reject(monkeypatch):
    app = _app()
    monkeypatch.setattr(app, "_modal_result", lambda screen, default: "reject")
    assert app.approve_write("write_file", {"file_path": "/n.txt"}) is False


def test_approve_write_auto_skips_later_prompts(monkeypatch):
    app = _app()
    calls: list[int] = []
    monkeypatch.setattr(
        app, "_modal_result", lambda screen, default: calls.append(1) or "auto"
    )
    assert app.approve_write("write_file", {"file_path": "/a.txt"}) is True
    assert app.approve_write("edit_file", {"file_path": "/b.txt"}) is True
    assert calls == [1]  # the second write was auto-approved without a modal
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run pytest tests/test_live_tui.py -k approve_write -q`
Expected: FAIL — `LiveAgentApp` has no attribute `approve_write`.

- [ ] **Step 3: Implement**

In `aai_cli/agent_cascade/tui.py`, add to the imports:

```python
import threading
```

and (with the other `code_agent` imports):

```python
from aai_cli.code_agent.modals import ApprovalScreen
```

Add the transparent-modal rule to the `CSS` block (so the bottom-docked modal shows the transcript above it, matching the code TUI):

```css
    /* The approval modal docks at the bottom and must stay see-through (the transcript shows
       above it), overriding ModalScreen's default opaque DEFAULT_CSS. */
    ModalScreen { background: transparent; }
```

In `__init__`, add the auto-approve latch (near `self._interrupt`):

```python
        self._auto_approve_writes = False  # set once the user picks "auto" on a write prompt
```

Add the approval methods (in the interrupt/quit section):

```python
    def _modal_result[T](self, screen: ModalScreen[T], default: T) -> T:
        """Push a modal from the cascade worker thread and block until it's dismissed."""
        done = threading.Event()
        box: dict[str, T] = {"value": default}

        def _store(result: T | None) -> None:
            if result is not None:
                box["value"] = result
            done.set()

        self.call_from_thread(self.push_screen, screen, _store)
        done.wait()
        return box["value"]

    def approve_write(self, name: str, args: dict[str, object]) -> bool:
        """Decide a gated write by a y/n keypress; True to allow.

        Called on the cascade worker thread (via the brain's approver). Blocks on a bottom-docked
        approval modal so the user confirms a file write by keyboard — the one place the
        hands-free session pauses for input. "Auto" approves every later write this session.
        """
        if self._auto_approve_writes:
            return True
        decision = self._modal_result(ApprovalScreen(name, args), default="reject")
        if decision == "auto":
            self._auto_approve_writes = True
            return True
        return decision == "approve"
```

Add the `ModalScreen` import to the typing block:

```python
from textual.screen import ModalScreen
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `uv run pytest tests/test_live_tui.py -k approve_write -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add aai_cli/agent_cascade/tui.py tests/test_live_tui.py
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): TUI write-approval modal reusing code agent's ApprovalScreen"
```

---

### Task 6: `--files` flag + command wiring (TUI approver, headless deny)

Expose the flag, carry it into `CascadeConfig`, wire the TUI's `approve_write` into the cascade deps, and auto-deny writes on the headless path.

**Files:**
- Modify: `aai_cli/commands/agent_cascade/__init__.py` (the `--files` option + epilog example)
- Modify: `aai_cli/commands/agent_cascade/_exec.py` (`AgentCascadeOptions.files`, config wiring, deps approver on both paths)
- Test: `tests/test_agent_cascade_command.py`, `tests/test_live_tui.py`
- Snapshot: `tests/__snapshots__/test_snapshots_help_run.ambr` (regenerated)

**Interfaces:**
- Consumes: `CascadeConfig.files` (Task 1), `CascadeDeps.real(..., approver=...)` (Task 4), `LiveAgentApp.approve_write` (Task 5).
- Produces: `AgentCascadeOptions.files: bool`; `_exec._deny_writes(name, args) -> bool` (always `False`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_agent_cascade_command.py` (the `_opts` helper builds an `AgentCascadeOptions`; add `files=False` to its defaults dict so existing callers stay valid — see Step 3):

```python
def test_files_flag_flows_into_config(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(_exec.tts_session, "require_available", lambda _c: None)
    monkeypatch.setattr(config, "resolve_api_key", lambda **_: "k")
    monkeypatch.setattr(_exec, "FileSource", lambda src: types.SimpleNamespace(sample_rate=16000))
    monkeypatch.setattr(_exec.client, "resolve_audio_source", lambda source, sample: "clip.wav")
    monkeypatch.setattr(_exec, "_should_use_tui", lambda **_: False)
    monkeypatch.setattr(_exec, "_warn_without_web_search", lambda **_: None)
    monkeypatch.setattr(
        _exec, "_open_audio", lambda *a, **k: (iter([]), _exec.NullPlayer(), 16000)
    )

    def fake_real(api_key, cfg, *, audio, stt_params, approver=None):
        captured["files"] = cfg.files
        captured["approver"] = approver
        return _exec.engine.CascadeDeps(
            run_stt=lambda on_turn: None, complete_reply=lambda m, on_tool=None: "", synthesize=lambda t: b""
        )

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kwargs: None)
    run_agent_cascade(_opts(source="clip.wav", files=True), _state(), json_mode=False)
    assert captured["files"] is True
    # Headless path: writes can't be confirmed, so the approver denies them.
    assert captured["approver"]("write_file", {}) is False


def test_deny_writes_always_false():
    assert _exec._deny_writes("write_file", {"file_path": "/x"}) is False
```

(Use the file's existing `_state()`/`AppState` helper; if absent, mirror the `AppState` construction in `test_run_wires_deps_and_invokes_cascade`.)

In `tests/test_live_tui.py`, assert the TUI path wires the app's approver:

```python
def test_tui_wires_app_approver(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(_exec, "DuplexAudio", lambda **k: types.SimpleNamespace(
        mic=iter([]), player=_exec.NullPlayer(), close=lambda: None, toggle_listening=lambda: True
    ))
    monkeypatch.setattr(_exec, "_build_stt_params", lambda opts, rate: object())

    def fake_real(api_key, cfg, *, audio, stt_params, approver=None):
        captured["approver"] = approver
        return _exec.engine.CascadeDeps(
            run_stt=lambda on_turn: None, complete_reply=lambda m, on_tool=None: "", synthesize=lambda t: b""
        )

    monkeypatch.setattr(_exec.engine.CascadeDeps, "real", fake_real)
    monkeypatch.setattr(_exec.engine, "run_cascade", lambda **kwargs: None)

    class _DummyApp:
        def __init__(self, **kwargs):
            self.approve_write = lambda name, args: True
        def run(self, **kwargs):
            pass
        error = None

    monkeypatch.setattr("aai_cli.agent_cascade.tui.LiveAgentApp", _DummyApp)
    _exec._run_live_tui("k", _opts(files=True), CascadeConfig(files=True))
    assert captured["approver"] is not None and captured["approver"]("write_file", {}) is True
```

- [ ] **Step 2: Run them, verify they fail**

Run: `uv run pytest tests/test_agent_cascade_command.py -k "files_flag or deny_writes" tests/test_live_tui.py -k tui_wires_app_approver -q`
Expected: FAIL — `AgentCascadeOptions` has no `files` / `_exec` has no `_deny_writes`.

- [ ] **Step 3: Implement the flag + wiring**

In `aai_cli/commands/agent_cascade/__init__.py`, add the option to `live(...)` (in the Tools panel, after `mcp_config`):

```python
    files: bool = typer.Option(
        False,
        "--files",
        help="Let the agent read and write files in the current directory (writes need y/n confirmation)",
        rich_help_panel=_PANEL_TOOLS,
    ),
```

Add an epilog example (in the `examples_epilog([...])` list):

```python
            (
                "Let the agent read and write files here",
                "assembly --sandbox live --files",
            ),
```

Pass it into the options constructor:

```python
        mcp_config=tuple(mcp_config or ()),
        files=files,
        show_code=show_code,
```

In `aai_cli/commands/agent_cascade/_exec.py`, add the dataclass field (after `mcp_config`):

```python
    # Let the agent read/write files in the launch directory (writes confirmed; none by default).
    files: bool
```

Add the headless deny approver (module level, near `_web_search_note`):

```python
def _deny_writes(name: str, args: dict[str, object]) -> bool:
    """Approver for non-interactive runs: deny every gated write (no channel to confirm one).

    Reads stay ungated (they never reach an approver), so a piped/file/--json `--files` session
    can still read and search — it just can't write without a TUI to press y/n in.
    """
    del name, args
    return False
```

Set `files=opts.files` on **both** `CascadeConfig(...)` constructions (the live config in `run_agent_cascade` and the `_print_show_code` config). For `run_agent_cascade`'s config add:

```python
        mcp_servers=mcp_servers,
        files=opts.files,
```

For `_print_show_code`'s config add `files=opts.files,` (so `--show-code` reflects the flag in the constructed config even though the generated script is unaffected — see Step 5).

Pass the deny approver on the headless path — in `run_agent_cascade`, change the `CascadeDeps.real(...)` call:

```python
    deps = engine.CascadeDeps.real(
        api_key, config, audio=audio, stt_params=stt_params, approver=_deny_writes
    )
```

Wire the TUI's approver — in `_run_live_tui`, build the app first, then the deps referencing `app.approve_write` (the closure resolves `app` at call time, after it's assigned):

```python
def _run_live_tui(api_key: str, opts: AgentCascadeOptions, config: CascadeConfig) -> None:
    from aai_cli.agent_cascade.tui import LiveAgentApp

    duplex = DuplexAudio(target_rate=SAMPLE_RATE, device=opts.device)
    stt_params = _build_stt_params(opts, SAMPLE_RATE)

    def approve_write(name: str, args: dict[str, object]) -> bool:
        return app.approve_write(name, args)

    deps = engine.CascadeDeps.real(
        api_key, config, audio=duplex.mic, stt_params=stt_params, approver=approve_write
    )

    def run_conversation(renderer: engine.Renderer) -> None:
        engine.run_cascade(
            renderer=renderer,
            player=duplex.player,
            config=config,
            deps=deps,
            on_session=lambda session: app.set_interrupt(session.interrupt_reply),
        )

    app = LiveAgentApp(
        run_conversation=run_conversation,
        on_stop=duplex.close,
        on_toggle_listen=duplex.toggle_listening,
        web_note=_web_search_note(),
    )
    app.run(mouse=False)
    if app.error is not None:
        raise app.error
```

Update the `_opts` helper in `tests/test_agent_cascade_command.py` to include `files=False` in its defaults, and update any existing test that constructs `CascadeDeps.real(...)` or a `fake_real`/`fake_build_completer` to accept the new keyword-only `approver` parameter (e.g. the `fake_real` at ~line 230 and the `build_completer` lambda at ~line 248 — give them `*, approver=None`).

- [ ] **Step 4: Run the targeted tests, verify they pass**

Run: `uv run pytest tests/test_agent_cascade_command.py tests/test_live_tui.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm `--show-code` is unaffected, then regenerate the help snapshot**

The generated script (`code_gen.agent_cascade`) models the STT→LLM→TTS SDK cascade, not the deepagents toolset, so `--files` does not change its output — no code-gen change is needed (the flag is carried on the config only for completeness). Confirm:

Run: `uv run pytest tests/test_code_gen_agent_cascade.py tests/test_agent_cascade_show_code.py -q`
Expected: PASS.

Regenerate the `live --help` golden (the new `--files` row + epilog example land here):

Run: `uv run pytest tests/test_snapshots_help_run.py --snapshot-update -q`
Then verify clean: `uv run pytest tests/test_snapshots_help_run.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add aai_cli/commands/agent_cascade/__init__.py aai_cli/commands/agent_cascade/_exec.py tests/test_agent_cascade_command.py tests/test_live_tui.py tests/__snapshots__/test_snapshots_help_run.ambr
AAI_ALLOW_COMMIT=1 git commit -m "feat(live): --files flag wiring (TUI approver + headless deny)"
```

---

### Task 7: Docs consistency + full gate

Document the flag and run the authoritative gate to green.

**Files:**
- Modify: `REFERENCE.md`, `README.md` (if either enumerates `assembly live` flags — the docs consistency gate checks `assembly …` command refs)
- Modify: `aai_cli/AGENTS.md` (note the new capability under the `agent_cascade/` subsystem bullet)

- [ ] **Step 1: Check what the docs gate expects**

Run: `uv run python scripts/docs_consistency_gate.py`
Expected: PASS, or a specific instruction about a missing `assembly live --files` reference / env var. Fix exactly what it reports (add a `--files` mention to the `live` section of `REFERENCE.md`/`README.md` if flagged).

- [ ] **Step 2: Add a one-line note to `aai_cli/AGENTS.md`**

In the `agent_cascade/` subsystem bullet, append a sentence:

```
`--files` swaps the brain's in-memory backend for a real-cwd `FilesystemBackend`
(deepagents) and gates `write_file`/`edit_file` behind a TUI `y/n` approval (the
code agent's `ApprovalScreen`); reads (incl. `grep`) stay ungated and headless runs
auto-deny writes.
```

- [ ] **Step 3: Run the full gate**

Run: `./scripts/check.sh`
Expected: ends with `All checks passed.` Fix any mutation-gate survivors on changed lines by tightening assertions (the diff-scoped mutation + 100% patch-coverage stages are the ones most likely to flag gaps). Re-run until green.

- [ ] **Step 4: Final commit (no override — the gate just passed)**

```bash
git add REFERENCE.md README.md aai_cli/AGENTS.md
git commit -m "docs(live): document --files file read/write capability"
```

---

## Self-Review

**Spec coverage:**
- Opt-in flag default off → Task 1 (config) + Task 6 (flag). ✓
- Reads ungated incl. grep; writes gated → Task 1 (`interrupt_on` only write tools) + Task 2 (grep label). ✓
- TUI keypress confirm → Task 5; headless auto-deny → Task 6 (`_deny_writes`). ✓
- `FilesystemBackend` rooted at cwd, virtual_mode → Task 1. ✓
- `execute` inert/unadvertised/ungated → Task 1 (only `_WRITE_TOOLS` gated) + Task 2 (not in capability phrase). ✓
- Reply timeout excludes approval wait → Task 4. ✓
- Capability advertised → Task 2. ✓
- `--show-code` unaffected (verified) → Task 6 Step 5. ✓
- Tests for brain/engine/TUI → Tasks 1–6. ✓
- Docs consistency → Task 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows the code; commands have expected output.

**Type consistency:** `Approver = Callable[[str, dict[str, object]], bool]` used identically in `brain.build_completer`, `engine.CascadeDeps.real`, `_exec._deny_writes`, and `LiveAgentApp.approve_write`. `_complete_within(messages, timeout: float | None)` matches its two call sites (`None` when `config.files`, else `_REPLY_TIMEOUT_SECONDS`). `_graph_kwargs`/`_build_fs_backend` names match between definition (Task 1) and use (`build_graph`). `CascadeConfig.files` consistent across config/brain/engine/exec.
