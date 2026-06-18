"""Bottom-docked modal screens for the coding-agent TUI: tool approval and agent questions.

Split out of `tui.py` to keep each module under the file-length gate. Both are transparent
``ModalScreen``s docked at the bottom, so the transcript stays visible above them (see the
``ModalScreen { background: transparent }`` rule in :class:`~aai_cli.code_agent.tui.CodeAgentApp`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label

from aai_cli.code_agent import banner, risk
from aai_cli.code_agent.summarize import describe_args, full_args

if TYPE_CHECKING:
    from collections.abc import Mapping


class ApprovalScreen(ModalScreen[str]):
    """A compact, bottom-docked prompt to approve/auto-approve/reject one tool call.

    Keyboard-only — a plain one-line ``y / a / n`` hint instead of clickable buttons, so it
    reads like a CLI prompt rather than a chrome-heavy dialog. The transparent screen
    background leaves the transcript visible above (no full-screen takeover); the decision is
    one of ``"approve"``, ``"auto"``, or ``"reject"``. The compact arg view expands to the full
    args on ``e`` (deepagents-code's collapse/expand), and a risky call (``rm -rf``, a fetch of
    an internal address) carries a one-line warning above the prompt.
    """

    DEFAULT_CSS = """
    ApprovalScreen { align: center bottom; background: transparent; }
    ApprovalScreen #approvalbox {
        dock: bottom; width: 1fr; height: auto;
        border: round #f59e0b; background: #000000; padding: 0 1; margin: 0 1 1 1;
    }
    ApprovalScreen #approvalbox Label { height: auto; }
    """
    BINDINGS: ClassVar = [
        ("y", "approve", "Approve"),
        ("a", "auto", "Auto-approve"),
        ("n", "reject", "Reject"),
        ("e", "expand", "Expand"),
    ]

    def __init__(self, name: str, args: Mapping[str, object]) -> None:
        super().__init__()
        self._tool_name = name  # not _name: that shadows Textual Widget's str|None attr
        self._args = args
        self._expanded = False  # toggled by `e`; collapsed (one-line) by default

    def compose(self) -> ComposeResult:
        with Vertical(id="approvalbox"):
            warning = risk.risk_warning(self._tool_name, self._args)
            if warning:
                yield Label(f"[b #f04438]⚠ {escape(warning)}[/]", id="approvalwarn")
            yield Label(self._detail_markup(), id="approvaldetail")
            yield Label(
                f"[b #22c55e]y[/] approve   [b {banner.BRAND_HEX}]a[/] auto-approve   "
                "[b #f04438]n[/] reject   [b]e[/] expand"
            )

    def _detail_markup(self) -> str:
        """The 'Run tool X?' line — the compact arg, or the full args when expanded."""
        args = full_args(self._args) if self._expanded else describe_args(self._args)
        return f"Run tool [b]{escape(self._tool_name)}[/b]?  [dim]{escape(args)}[/dim]"

    def action_expand(self) -> None:
        """Toggle between the compact identifying arg and the full args (``e``)."""
        self._expanded = not self._expanded
        self.query_one("#approvaldetail", Label).update(self._detail_markup())

    def action_approve(self) -> None:
        self.dismiss("approve")

    def action_auto(self) -> None:
        self.dismiss("auto")

    def action_reject(self) -> None:
        self.dismiss("reject")


class AskScreen(ModalScreen[str]):
    """A bottom-docked prompt that relays a question from the agent and returns the answer."""

    DEFAULT_CSS = """
    AskScreen { align: center bottom; background: transparent; }
    AskScreen #askbox {
        dock: bottom; width: 1fr; height: auto;
        border: round #3a3f55; background: #000000; padding: 0 1; margin: 0 1 1 1;
    }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self._question = question

    def compose(self) -> ComposeResult:
        with Vertical(id="askbox"):
            yield Label(f"[b]The agent asks:[/b] {escape(self._question)}")
            yield Input(id="answer", placeholder="Type your answer and press Enter…")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)
