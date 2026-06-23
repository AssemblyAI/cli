"""Bottom-docked approval modal for the live voice agent TUI.

Split out of ``tui.py`` to keep each module under the file-length gate. The
``ApprovalScreen`` is a transparent ``ModalScreen`` docked at the bottom, so the
transcript stays visible above it (see the ``ModalScreen { background: transparent }``
rule in :class:`~aai_cli.agent_cascade.tui.LiveAgentApp`).

The keyboard path (``y / a / n / e``) is always available; under ``--files`` an open modal
can *also* be resolved by voice (:meth:`ApprovalScreen.try_voice`), the engine routing the
next spoken transcript here so the gate stays hands-free — except destructive commands, which
ignore the spoken answer and require a keypress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label

from aai_cli.agent_cascade import banner, risk
from aai_cli.agent_cascade.spoken_approval import spoken_decision
from aai_cli.agent_cascade.summarize import describe_args, full_args

if TYPE_CHECKING:
    from collections.abc import Mapping


class ApprovalScreen(ModalScreen[str]):
    """A compact, bottom-docked prompt to approve/auto-approve/reject one tool call.

    Keyboard ``y / a / n`` (and ``e`` to expand the args). The transparent background
    leaves the transcript visible, and a risky call (``rm -rf``, an internal fetch)
    carries a warning.
    """

    DEFAULT_CSS = """
    ApprovalScreen { align: center bottom; background: transparent; }
    /* width: 100% (not 1fr) so the box honors its 1-col side margins — a docked 1fr container
       ignores horizontal margin and overflows the screen, clipping the right border off-edge. */
    ApprovalScreen #approvalbox {
        dock: bottom; width: 100%; height: auto;
        border: round #f59e0b; background: #000000; padding: 0 1; margin: 0 1 1 1;
    }
    ApprovalScreen #approvalbox Label { height: auto; }
    """
    BINDINGS: ClassVar = [
        ("y", "approve", "Approve"),
        ("a", "auto", "Auto-approve"),
        ("n", "reject", "Reject"),
        ("e", "expand", "Expand"),
        # Escape / Ctrl-C dismiss the modal — declining the tool is the safe cancel.
        ("escape,ctrl+c", "reject", "Cancel"),
    ]

    def __init__(self, name: str, args: Mapping[str, object]) -> None:
        super().__init__()
        self._tool_name = name  # not _name: that shadows Textual Widget's str|None attr
        self._args = args
        self._expanded = False  # toggled by `e`; collapsed (one-line) by default
        # Must start False so the first y/a/n decision dismisses; pinned by
        # test_approval_screen_starts_unanswered (and the keyboard pilots). pragma: the mutation
        # harness mis-selects covering tests for this Textual __init__ line (false survivor).
        self._answered = False  # pragma: no mutate — guards against a double dismiss

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

    def _decide(self, decision: str) -> None:
        """Dismiss once, whether the answer came by keypress or voice."""
        if self._answered:
            return
        self._answered = True
        self.dismiss(decision)

    def try_voice(self, transcript: str) -> None:
        """Resolve this open modal from a spoken transcript (the hands-free path).

        Destructive commands ignore the spoken answer (``spoken_decision`` returns None) so only
        a keypress can green-light them; otherwise an unambiguous affirmative approves and
        anything else (negation, bare "yes", unrelated speech) rejects — fail-safe. A no-op once
        already answered (a keypress won the race)."""
        decision = spoken_decision(self._tool_name, self._args, transcript)
        if decision is None:
            return
        self._decide("approve" if decision else "reject")

    def _detail_markup(self) -> str:
        """The 'Run tool X?' line — the compact arg, or the full args when expanded."""
        args = full_args(self._args) if self._expanded else describe_args(self._args)
        return f"Run tool [b]{escape(self._tool_name)}[/b]?  [dim]{escape(args)}[/dim]"

    def action_expand(self) -> None:
        """Toggle between the compact identifying arg and the full args (``e``)."""
        self._expanded = not self._expanded
        self.query_one("#approvaldetail", Label).update(self._detail_markup())

    def action_approve(self) -> None:
        self._decide("approve")

    def action_auto(self) -> None:
        self._decide("auto")

    def action_reject(self) -> None:
        self._decide("reject")
