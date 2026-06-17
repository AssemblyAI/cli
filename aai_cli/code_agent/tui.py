"""A Textual terminal UI for the coding agent, modeled on deepagents-code.

deepagents' own `code` CLI is a Textual app (a scrolling conversation transcript with
a bottom input and modal tool-approval prompts); this mirrors that design on top of
the same :class:`~aai_cli.code_agent.session.CodeSession`. The agent runs on a thread
worker (its `invoke` is synchronous), streaming display events back onto the UI thread
via ``call_from_thread``; tool approvals pause the worker on a modal screen.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RichLog, Static
from textual.worker import Worker

from aai_cli.code_agent import banner
from aai_cli.code_agent.agent import CompiledAgent
from aai_cli.code_agent.ask_tool import AskBridge
from aai_cli.code_agent.events import AssistantText, ErrorText, Event, ToolCall, ToolResult
from aai_cli.code_agent.session import CodeSession

if TYPE_CHECKING:
    from collections.abc import Mapping


def _format_args(args: Mapping[str, object]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in args.items())


def _abbrev_home(path: Path) -> str:
    """Render ``path`` with the home directory collapsed to ``~``."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _git_branch(start: Path) -> str | None:
    """The current git branch for ``start`` (walking up to the repo root), or None."""
    for directory in (start, *start.parents):
        head = directory / ".git" / "HEAD"
        if head.is_file():
            ref = head.read_text(encoding="utf-8").strip()
            return ref.removeprefix("ref: refs/heads/") if ref.startswith("ref: ") else ref[:8]
    return None


def _status_text(cwd: Path, *, auto_approve: bool) -> str:
    """The bottom status line: a mode badge, the working directory, and the git branch."""
    mode = "auto" if auto_approve else "manual"
    badge = f"[black on #f59e0b] {mode} [/]"
    parts = [badge, f"[dim]{_abbrev_home(cwd)}[/dim]"]
    branch = _git_branch(cwd)
    if branch:
        parts.append(f"[dim]↗ {branch}[/dim]")
    return " ".join(parts)


class ApprovalScreen(ModalScreen[bool]):
    """A modal asking the user to approve or reject one pending tool call."""

    BINDINGS: ClassVar = [("y", "approve", "Approve"), ("n", "reject", "Reject")]

    def __init__(self, name: str, args: Mapping[str, object]) -> None:
        super().__init__()
        self._name = name
        self._args = args

    def compose(self) -> ComposeResult:
        yield Label(f"Run tool [b]{self._name}[/b]?\n{_format_args(self._args)}")
        with Horizontal():
            yield Button("Approve (y)", id="approve", variant="success")
            yield Button("Reject (n)", id="reject", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        approved = event.button.id == "approve"
        self.dismiss(approved)

    def action_approve(self) -> None:
        self.dismiss(result=True)

    def action_reject(self) -> None:
        self.dismiss(result=False)


class AskScreen(ModalScreen[str]):
    """A modal that asks the user a question from the agent and returns their answer."""

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        yield Label(f"[b]The agent asks:[/b]\n{self._question}")
        yield Input(id="answer", placeholder="Type your answer and press Enter…")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)


class CodeAgentApp(App[None]):
    """The coding-agent TUI: conversation transcript + prompt + approval/ask modals."""

    # Flat dark canvas — no panel borders/gray, just the bordered prompt and a status
    # line, matching the deepagents-code look (wordmark in the AssemblyAI brand blue).
    CSS = f"""
    Screen {{ background: #0b0e16; }}
    #log {{
        height: 1fr; border: none; background: #0b0e16; padding: 1 2;
        scrollbar-size-vertical: 0;
    }}
    #promptbar {{ dock: bottom; height: 3; background: #0b0e16; border: round #3a3f55; margin: 1 1; }}
    #promptmark {{ width: 3; color: {banner.BRAND_HEX}; content-align: center middle; }}
    #prompt {{ border: none; background: #0b0e16; padding: 0; }}
    #status {{ dock: bottom; height: 1; background: #0b0e16; padding: 0 1; }}
    """
    TITLE = "AssemblyAI Code"
    # Ctrl-C quits (in addition to Ctrl-Q); the built-in command palette is removed.
    ENABLE_COMMAND_PALETTE = False
    BINDINGS: ClassVar = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+y", "copy_last", "Copy last reply"),
    ]

    def __init__(
        self,
        *,
        agent: CompiledAgent,
        ask_bridge: AskBridge | None = None,
        auto_approve: bool = False,
        initial: str | None = None,
        thread_id: str = "default",
        cwd: Path | None = None,
        web_note: str | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._ask_bridge = ask_bridge if ask_bridge is not None else AskBridge()
        self._auto_approve = auto_approve
        self._initial = initial
        self._session_name = thread_id  # not _thread_id: that shadows Textual App's int
        self._cwd = cwd if cwd is not None else Path.cwd()
        self._web_note = web_note
        self._last_reply = ""
        self._session = CodeSession(
            agent=agent,
            sink=self._emit_event,
            approver=self._approve,
            thread_id=thread_id,
            auto_approve=auto_approve,
        )

    def compose(self) -> ComposeResult:
        # No Header/Footer chrome — the splash is the title and the bottom status line
        # the only footer, so the screen stays a flat dark canvas.
        yield RichLog(id="log", wrap=True, markup=True)
        with Horizontal(id="promptbar"):
            yield Static(">", id="promptmark")
            yield Input(id="prompt", placeholder="Ask the agent to build something…")
        yield Static(_status_text(self._cwd, auto_approve=self._auto_approve), id="status")

    def _write_splash(self, log: RichLog) -> None:
        for row in banner.wordmark():
            log.write(f"[bold {banner.BRAND_HEX}]{row}[/]")
        log.write(f"[dim]{banner.version()}[/dim]")
        log.write("")
        log.write(f"[dim]Thread: {self._session_name}[/dim]")
        log.write("")
        log.write(f"[{banner.BRAND_HEX}]{banner.READY_LINE}[/]")
        log.write(f"[dim]{banner.TIP_LINE}[/dim]")

    def on_mount(self) -> None:
        # Route the agent's ask_user tool through a modal (the bridge is shared with
        # the tool built before this app existed).
        self._ask_bridge.handler = self._ask
        self._write_splash(self.query_one("#log", RichLog))
        if self._web_note:
            self.notify(self._web_note, title="Web search disabled", severity="warning", timeout=10)
        # Put the cursor in the prompt so the user can type immediately (RichLog would
        # otherwise hold focus and swallow keystrokes).
        self.query_one("#prompt", Input).focus()
        if self._initial:
            self._submit(self._initial)

    # --- event rendering (always called on the UI thread) ---------------------

    def _emit_event(self, event: Event) -> None:
        """Sink for :class:`CodeSession`; marshaled onto the UI thread by the worker."""
        self.call_from_thread(self._write_event, event)

    def _write_event(self, event: Event) -> None:
        log = self.query_one("#log", RichLog)
        if isinstance(event, AssistantText):
            self._last_reply = event.text
            log.write(event.text)
        elif isinstance(event, ToolCall):
            log.write(f"[dim]→ {event.name}({_format_args(event.args)})[/dim]")
        elif isinstance(event, ToolResult):
            log.write(f"[dim]  {event.name}: {event.content.strip()[:2000]}[/dim]")
        elif isinstance(event, ErrorText):
            log.write(f"[#F04438]✗ {event.text}[/#F04438]")

    def action_copy_last(self) -> None:
        """Copy the most recent assistant reply to the system clipboard."""
        import pyperclip

        if self._last_reply:
            pyperclip.copy(self._last_reply)
            self.query_one("#log", RichLog).write("[dim](copied last reply to clipboard)[/dim]")

    # --- approval / ask (called on the worker thread) -------------------------

    def _modal_result[T](self, screen: ModalScreen[T], default: T) -> T:
        """Push a modal from the worker thread and block until it's dismissed."""
        done = threading.Event()
        box: dict[str, T] = {"value": default}

        def _store(result: T | None) -> None:
            if result is not None:
                box["value"] = result
            done.set()

        self.call_from_thread(self.push_screen, screen, _store)
        done.wait()
        return box["value"]

    def _approve(self, name: str, args: dict[str, object]) -> bool:
        """Block the worker on a modal approval screen and return the decision."""
        return self._modal_result(ApprovalScreen(name, args), default=False)

    def _ask(self, question: str) -> str:
        """Block the worker on a modal input screen and return the user's answer."""
        return self._modal_result(AskScreen(question), default="")

    # --- input loop -----------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self._submit(text)

    def _submit(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        log.write(f"[b cyan]» {text}[/b cyan]")
        self.query_one("#prompt", Input).disabled = True
        self._run_turn(text)

    def _run_turn(self, text: str) -> Worker[None]:
        return self.run_worker(
            lambda: self._session.send(text), thread=True, exclusive=True, name="agent-turn"
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.is_finished:
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()
