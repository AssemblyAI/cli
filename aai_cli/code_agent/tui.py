"""A Textual terminal UI for the coding agent, modeled on deepagents-code.

deepagents' own `code` CLI is a Textual app (a scrolling conversation transcript with
a bottom input and modal tool-approval prompts); this mirrors that design on top of
the same :class:`~aai_cli.code_agent.session.CodeSession`. The agent runs on a thread
worker (its `invoke` is synchronous), streaming display events back onto the UI thread
via ``call_from_thread``; tool approvals pause the worker on a modal screen.
"""

from __future__ import annotations

import itertools
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Static
from textual.worker import Worker

from aai_cli.code_agent import banner
from aai_cli.code_agent.agent import CompiledAgent
from aai_cli.code_agent.ask_tool import AskBridge
from aai_cli.code_agent.events import (
    AssistantDelta,
    AssistantText,
    ErrorText,
    Event,
    ToolCall,
    ToolResult,
)
from aai_cli.code_agent.messages import (
    AssistantMessage,
    ErrorMessage,
    Note,
    ToolCallLine,
    ToolOutput,
    UserMessage,
)
from aai_cli.code_agent.modals import ApprovalScreen, AskScreen
from aai_cli.code_agent.session import CodeSession
from aai_cli.code_agent.tui_status import (
    VOICE_FRAMES,
    _spinner_text,
    _status_text,
    voicebar_markup,
)
from aai_cli.code_agent.voice_ui import _VoiceIO, _VoiceLegs

if TYPE_CHECKING:
    from textual.timer import Timer

# Glyphs cycled by the working indicator's animation (purely cosmetic).
_SPIN_FRAMES = "✶✷✸✹✺"  # pragma: no mutate
# Seconds the Ctrl-C "press again to quit" hint stays armed (deepagents-code uses 3s too).
_QUIT_HINT_SECONDS = 3  # pragma: no mutate


class CodeAgentApp(_VoiceLegs):
    """The coding-agent TUI: conversation transcript + prompt + approval/ask modals."""

    # Flat pure-black canvas — no panel fills/gray, just the bordered prompt and a status
    # line, matching the deepagents-code look (wordmark in the AssemblyAI brand blue).
    CSS = f"""
    Screen {{ background: #000000; }}
    /* The approval/ask modals must stay see-through so the transcript shows above their
       docked prompt. Their own DEFAULT_CSS sets `background: transparent`, but app CSS beats
       a widget's DEFAULT_CSS — without this rule the `Screen` canvas above paints the modal
       opaque black (it matches every Screen subclass) and blanks the transcript behind it. */
    ModalScreen {{ background: transparent; }}
    /* The transcript is a scroll container of mounted message widgets (not a RichLog), so the
       reply streams in place and tool output can expand/collapse. */
    #log {{ height: 1fr; border: none; background: #000000; padding: 1 2; }}
    /* width: 100% (not the 1fr default) so the bordered box fits inside its 1-col side margins;
       a docked 1fr container ignores horizontal margin and overflows, clipping the right border. */
    #promptbar {{ dock: bottom; height: 3; width: 100%; background: #000000; border: round #3a3f55; margin: 1 1; }}
    #promptmark {{ width: 3; color: {banner.BRAND_HEX}; content-align: center middle; }}
    #prompt {{ border: none; background: #000000; padding: 0; }}
    /* Shown in place of the prompt while voice capture is on (Ctrl-V brings the prompt back). */
    #voicebar {{ dock: bottom; height: 3; background: #000000; border: round {banner.BRAND_HEX};
        margin: 1 1; content-align: center middle; display: none; }}
    /* In normal flow below the 1fr log, so it sits just above the docked prompt bar. */
    #spinner {{ height: 1; background: #000000; padding: 0 2;
        color: {banner.BRAND_HEX}; display: none; }}
    #status {{ dock: bottom; height: 1; background: #000000; padding: 0 1; }}
    """
    TITLE = "AssemblyAI Code"
    # Ctrl-C quits (in addition to Ctrl-Q); the built-in command palette is removed.
    ENABLE_COMMAND_PALETTE = False
    # Interrupt/quit keys follow deepagents-code: Escape interrupts the running turn (or, in
    # voice mode, the active listen/readback), and Ctrl-C interrupts a running turn or active
    # voice, or — when idle — quits only on a confirmed double-press.
    BINDINGS: ClassVar = [
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+c", "quit_or_interrupt", "Interrupt / Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+y", "copy_last", "Copy last reply"),
        ("ctrl+v", "toggle_voice", "Toggle voice"),
        ("ctrl+o", "toggle_output", "Expand/collapse output"),
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
        voice: _VoiceIO | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._ask_bridge = ask_bridge if ask_bridge is not None else AskBridge()
        self._auto_approve = auto_approve
        self._initial = initial
        self._voice = voice  # when set, spoken turns drive the prompt and replies are read back
        self._voice_typed = False  # flips once the mic is ruled out; then input is typed only
        self._voice_paused = False  # user-toggled off via Ctrl-V (distinct from a mic failure)
        self._voice_phase = "listening"  # listening / thinking / speaking, shown in the voice bar
        self._voice_frames = itertools.cycle(VOICE_FRAMES)
        self._voice_timer: Timer | None = None  # animates the voice-bar meter while it's shown
        self._streaming_msg: AssistantMessage | None = None  # the reply widget tokens stream into
        self._last_tool_output: ToolOutput | None = None  # the row Ctrl+O expands/collapses
        self._session_name = thread_id  # not _thread_id: that shadows Textual App's int
        self._cwd = cwd if cwd is not None else Path.cwd()
        self._web_note = web_note
        self._last_reply = ""
        self._quit_pending = False  # armed by a first idle Ctrl-C; a second confirms quit
        self._spin_frames = itertools.cycle(_SPIN_FRAMES)
        self._spin_timer: Timer | None = None
        self._turn_started = 0.0  # pragma: no mutate — always reset by _start_spinner first
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
        yield VerticalScroll(id="log")
        # Docked before the prompt bar, so the working indicator sits just above the input.
        yield Static("", id="spinner")
        with Horizontal(id="promptbar"):
            yield Static(">", id="promptmark")
            yield Input(id="prompt", placeholder="Ask the agent to build something…")
        yield Static("", id="voicebar")  # filled by _render_voicebar when voice mode is shown
        yield Static(
            _status_text(
                self._cwd, auto_approve=self._auto_approve, voice_state=self._voice_state()
            ),
            id="status",
        )

    def _write_splash(self) -> None:
        # The whole splash is fixed copy except the session name, so this markup is safe to
        # parse (only the session name — a --session value — is escaped).
        rows = [f"[bold {banner.BRAND_HEX}]{row}[/]" for row in banner.wordmark()]
        rows += [
            f"[dim]{banner.version()}[/dim]",
            "",
            f"[dim]Thread: {escape(self._session_name)}[/dim]",
            "",
            f"[{banner.BRAND_HEX}]{banner.READY_LINE}[/]",
            f"[dim]{banner.TIP_LINE}[/dim]",
        ]
        self._mount("\n".join(rows))

    def _mount(self, widget: Static | str) -> None:
        """Append a transcript widget (or a markup string) and scroll it into view."""
        log = self.query_one("#log", VerticalScroll)
        log.mount(Static(widget) if isinstance(widget, str) else widget)
        log.scroll_end(animate=False)  # pragma: no mutate — cosmetic; animate flag is unassertable

    def _note(self, text: str) -> None:
        """Append a dim transcript aside (cancelling / copied / voice-off)."""
        self._mount(Note(text))

    def on_mount(self) -> None:
        # Route the agent's ask_user tool through a modal (the bridge is shared with
        # the tool built before this app existed).
        self._ask_bridge.handler = self._ask
        self._write_splash()
        if self._web_note:
            self.notify(self._web_note, title="Web search disabled", severity="warning", timeout=10)
        # Put the cursor in the prompt so the user can type immediately (RichLog would
        # otherwise hold focus and swallow keystrokes).
        self.query_one("#prompt", Input).focus()
        self._sync_input_mode()  # in voice mode, swap the prompt for the listening affordance
        if self._initial:
            self._submit(self._initial)
        else:
            # Defer the first mic open until *after* the splash has painted. Opening PortAudio
            # is a GIL-holding C call; run inline on mount it races Textual's initial render and
            # the banner never flushes — it stays blank until a resize/focus forces a full
            # repaint. call_after_refresh runs once the screen is on-screen, so the splash wins.
            self.call_after_refresh(self._begin_listening)  # in voice mode, capture first turn

    # --- event rendering (always called on the UI thread) ---------------------

    def _emit_event(self, event: Event) -> None:
        """Sink for :class:`CodeSession`; marshaled onto the UI thread by the worker."""
        self.call_from_thread(self._write_event, event)

    def _write_event(self, event: Event) -> None:
        if isinstance(event, AssistantDelta):
            # Stream the token into the live reply widget (mounting one on the first token),
            # updated in place until the authoritative AssistantText finalizes it below.
            if self._streaming_msg is None:
                self._streaming_msg = AssistantMessage()
                self._mount(self._streaming_msg)
            self._streaming_msg.stream(event.text)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)  # pragma: no mutate
        elif isinstance(event, AssistantText):
            self._last_reply = event.text  # keep the raw text for clipboard copy
            self._finalize_reply(event.text)
        elif isinstance(event, ToolCall):
            self._mount(ToolCallLine(event.name, event.args))
        elif isinstance(event, ToolResult):
            self._last_tool_output = ToolOutput(event.name, event.content)
            self._mount(self._last_tool_output)
        elif isinstance(event, ErrorText):
            self._mount(ErrorMessage(event.text))

    def _finalize_reply(self, text: str) -> None:
        """Commit the reply: finalize the streamed widget in place, or mount a fresh one."""
        if self._streaming_msg is not None:
            self._streaming_msg.finalize(text)
            self._streaming_msg = None
        else:
            msg = AssistantMessage()
            self._mount(msg)
            msg.finalize(text)

    def action_copy_last(self) -> None:
        """Copy the most recent assistant reply to the system clipboard."""
        import pyperclip

        if self._last_reply:
            pyperclip.copy(self._last_reply)
            self._note("(copied last reply to clipboard)")

    def action_toggle_output(self) -> None:
        """Ctrl-O: expand/collapse the most recent tool output (a no-op if there's none)."""
        if self._last_tool_output is not None:
            self._last_tool_output.toggle()

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
        """Decide whether to run a gated tool, prompting unless auto-approve is on.

        Once the user picks "Auto-approve", later tool calls skip the modal entirely —
        functionally the same as starting with ``--auto``.
        """
        if self._auto_approve:
            return True
        screen = ApprovalScreen(name, args, voice=self._modal_voice())
        decision = self._modal_result(screen, default="reject")
        if decision == "auto":
            self._enable_auto_approve()
            return True
        return decision == "approve"

    def _enable_auto_approve(self) -> None:
        """Switch the session to auto-approve and refresh the mode badge."""
        self._auto_approve = True
        self._session.auto_approve = True
        self.call_from_thread(self._refresh_status)

    def _refresh_status(self) -> None:
        """Re-render the bottom status line (e.g. after the mode flips to auto or voice toggles)."""
        self.query_one("#status", Static).update(
            _status_text(
                self._cwd, auto_approve=self._auto_approve, voice_state=self._voice_state()
            )
        )

    def _voice_state(self) -> str | None:
        """``"on"``/``"off"`` for the status badge, or ``None`` when voice isn't wired up."""
        if self._voice is None:
            return None
        return "on" if self._voice_active() else "off"

    def action_toggle_voice(self) -> None:
        """Ctrl-V: turn spoken input/readback on or off for the session.

        A no-op notice when no voice front-end exists (e.g. a piped/typed run). Re-enabling
        kicks off listening again unless a turn is mid-flight (the post-turn followup will).
        """
        if self._voice is None:
            self.notify("Voice isn't available in this session", severity="warning")
            return
        self._voice_paused = not self._voice_paused
        self._refresh_status()
        self._sync_input_mode()  # show/hide the text box vs. the listening affordance
        if self._voice_paused:
            self.notify("Voice off — type your request")
        elif not self._turn_running():
            self.notify("Voice on — listening")
            self._begin_listening()

    def _sync_input_mode(self) -> None:
        """Swap the text prompt for the 'listening' affordance while voice capture is active.

        The Input stays mounted either way (it still holds the spoken transcript and the
        turn-running ``disabled`` flag); only the bars' visibility flips. The prompt regains
        focus whenever it's the visible input.
        """
        listening = self._voice_active()
        self.query_one("#promptbar", Horizontal).display = not listening
        self.query_one("#voicebar", Static).display = listening
        if listening:
            self._render_voicebar()
            if self._voice_timer is None:  # animate the meter only while the bar is shown
                self._voice_timer = self.set_interval(0.3, self._tick_voice)  # pragma: no mutate
        else:
            if self._voice_timer is not None:
                self._voice_timer.stop()
                self._voice_timer = None
            self.query_one("#prompt", Input).focus()

    def _set_voice_phase(self, phase: str) -> None:
        """Switch the voice bar between listening / thinking / speaking and repaint it."""
        self._voice_phase = phase
        self._render_voicebar()

    def _render_voicebar(self) -> None:
        """Paint the voice bar for the current phase: an animated meter, label, and accent."""
        hint = "   [dim](Ctrl-V to type)[/dim]" if self._voice_phase == "listening" else ""
        meter = next(self._voice_frames)
        self.query_one("#voicebar", Static).update(
            voicebar_markup(self._voice_phase, meter, hint=hint)
        )

    def _tick_voice(self) -> None:
        """Advance the voice-bar meter one frame (the animation timer's callback)."""
        self._render_voicebar()

    def _ask(self, question: str) -> str:
        """Block the worker on a modal input screen and return the user's answer."""
        return self._modal_result(AskScreen(question, voice=self._modal_voice()), default="")

    def _modal_voice(self) -> _VoiceIO | None:
        """The voice IO to drive a modal by speech, or ``None`` when voice isn't active."""
        return self._voice if self._voice_active() else None

    # --- interrupt / quit -----------------------------------------------------
    # Mirrors deepagents-code: Escape interrupts a running turn; Ctrl-C interrupts a running
    # turn or, when idle, quits only on a confirmed double-press (so it never drops the
    # conversation by accident). Ctrl-Q stays an unconditional one-press quit.

    def _turn_running(self) -> bool:
        """Whether an agent turn is in flight (the prompt is disabled while one runs)."""
        return self.query_one("#prompt", Input).disabled

    def _cancel_turn(self) -> bool:
        """Ask the session to stop its agent loop if a turn is running; True if one was.

        Cooperative: the worker keeps running until the streaming loop sees the flag at
        the next step boundary, then finishes and re-enables the prompt — so we never kill
        the thread mid-step (which Textual can't do safely anyway).
        """
        if not self._turn_running():
            return False
        self._session.request_cancel()
        self._note("cancelling…")
        return True

    def _stop_voice_activity(self) -> None:
        """Stop in-flight voice (a no-op when none is active).

        Interrupting the readback (speaking) stops it and resumes listening — the cancelled
        speak() returns and the loop captures the next turn. Interrupting while listening
        pauses voice to the text prompt, after which a second press falls through to quit.
        """
        if self._voice is None or not self._voice_active():
            return
        self._voice.cancel()
        if self._voice_phase == "speaking":  # stop talking, stay in voice mode -> re-listen
            self._note("stopped — listening…")
            return
        self._voice_paused = True
        self._refresh_status()
        self._sync_input_mode()  # active leg stopped -> bring the text prompt back
        self._note("voice interrupted (Ctrl-V to talk again)")

    def action_interrupt(self) -> None:
        """Escape: interrupt a running agent turn or in-flight voice (a no-op when idle)."""
        if not self._cancel_turn():
            self._stop_voice_activity()

    def action_quit_or_interrupt(self) -> None:
        """Ctrl-C: interrupt a running turn or active voice, else quit on a second press."""
        if self._cancel_turn():
            self._quit_pending = False
            return
        # A second press always quits — checked before stopping voice so a spoken turn can
        # never trap you (the first press stops the readback and arms; the second exits).
        if self._quit_pending:
            self.exit()
            return
        self._stop_voice_activity()  # stop a readback/listen if one's active (a no-op otherwise)
        self._arm_quit_pending()

    def _arm_quit_pending(self) -> None:
        """Arm Ctrl-C double-press-to-quit, showing a hint that expires after a few seconds."""
        self._quit_pending = True
        self.notify("Press Ctrl-C again to quit", timeout=_QUIT_HINT_SECONDS)
        self.set_timer(_QUIT_HINT_SECONDS, self._clear_quit_pending)

    def _clear_quit_pending(self) -> None:
        self._quit_pending = False  # pragma: no mutate — timer-fired reset; timing-unassertable

    # --- input loop -----------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if text:
            self._submit(text)

    def _submit(self, text: str) -> None:
        self._mount(UserMessage(text))
        self.query_one("#prompt", Input).disabled = True
        self._set_voice_phase("thinking")  # voice bar reflects the turn (no-op when bar hidden)
        self._start_spinner()
        self._run_turn(text)

    def _run_turn(self, text: str) -> Worker[None]:
        return self.run_worker(
            lambda: self._session.send(text), thread=True, exclusive=True, name="agent-turn"
        )

    # --- working indicator (spinner + elapsed) --------------------------------

    def _start_spinner(self) -> None:
        """Show the working indicator and animate it while the turn runs.

        Skipped in voice mode — the voice bar already shows a "Thinking…" state, so a second
        spinner would just be redundant chrome.
        """
        self._turn_started = time.monotonic()
        if self._voice_active():
            return
        self.query_one("#spinner", Static).display = True
        self._tick()
        self._spin_timer = self.set_interval(0.25, self._tick)  # pragma: no mutate

    def _tick(self) -> None:
        """Advance the spinner one frame and refresh the elapsed-seconds readout."""
        elapsed = int(time.monotonic() - self._turn_started)
        self.query_one("#spinner", Static).update(_spinner_text(elapsed, next(self._spin_frames)))

    def _stop_spinner(self) -> None:
        """Stop the animation and hide the working indicator."""
        if self._spin_timer is not None:
            self._spin_timer.stop()
            self._spin_timer = None
        self.query_one("#spinner", Static).display = False

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        # is_running guard: a worker finishing after teardown would hit an unmounted DOM.
        if event.worker.is_finished and self.is_running:
            self._finish_turn()

    def _finish_turn(self) -> None:
        """Wind down a completed turn: stop the spinner, re-enable input, resume voice."""
        self._stop_spinner()
        if self._streaming_msg is not None:  # a cancelled generation: keep what streamed in
            self._finalize_reply(self._streaming_msg.text)
        self.query_one("#prompt", Input).disabled = False
        self._sync_input_mode()  # focus the prompt (text mode) or show the listening bar
        self._voice_followup()  # read a spoken summary back, then listen for the next turn

    # The off-thread voice legs (_voice_active, _begin_listening, _capture_voice_turn, …) are
    # inherited from _VoiceLegs; the render/toggle side stays above.
