"""A voice-only Textual UI for `assembly live` (the agent cascade).

Shares the chrome of the `assembly code` TUI — the flat dark canvas, the ASSEMBLY
wordmark splash, the animated voice bar, and the transcript message widgets — but drops
the text prompt: `live` is a hands-free spoken conversation, so there is nothing to type.

The cascade (Streaming STT -> LLM -> streaming TTS) is handed in as a blocking
``run_conversation`` driven on a worker thread; it streams transcript events back through a
:class:`_TuiRenderer` that hops each call onto the UI thread. The voice bar tracks the phase
(listening / thinking / speaking). A quit calls ``on_stop`` to close the audio, which ends the
mic iterator and unblocks that worker.
"""

from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from aai_cli.code_agent import banner, tui_status
from aai_cli.code_agent.messages import AssistantMessage, ErrorMessage, Note, UserMessage
from aai_cli.core.errors import CLIError

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.timer import Timer

    from aai_cli.agent_cascade.engine import Renderer

# Splash intro copy (the code agent's banner copy is code-specific, so `live` carries its own).
_READY_LINE = "Listening… start talking when you're ready."
_TIP_LINE = "Use headphones — the mic stays open while the agent speaks."
# The one-line footer: a hands-free session, so the controls are interrupt-and-quit.
_STATUS_LINE = "Esc/Ctrl-C to interrupt · Ctrl-Q to quit"


class _TuiRenderer:
    """Marshals cascade :class:`~aai_cli.agent_cascade.engine.Renderer` calls onto the UI thread.

    The cascade runs on a worker thread; every render call hops back via ``call_from_thread``.
    Once the app has torn down (a quit mid-turn) that call raises ``RuntimeError`` — the event is
    moot then, so it's dropped rather than surfaced as an unhandled worker-thread exception.
    """

    def __init__(self, app: LiveAgentApp) -> None:
        self._app = app

    def connected(self) -> None:
        self._dispatch(self._app.live_connected)

    def user_partial(self, text: str) -> None:
        self._dispatch(self._app.show_user_partial, text)

    def user_final(self, text: str) -> None:
        self._dispatch(self._app.show_user_final, text)

    def reply_started(self) -> None:
        self._dispatch(self._app.begin_reply)

    def agent_transcript(self, text: str, *, interrupted: bool) -> None:
        # Sentences are emitted before any barge-in check, so `interrupted` is always False
        # here (the interrupted state is surfaced on reply_done); accept it for the protocol.
        del interrupted  # pragma: no mutate
        self._dispatch(self._app.show_agent_sentence, text)

    def reply_done(self, *, interrupted: bool) -> None:
        self._dispatch(lambda: self._app.end_reply(interrupted=interrupted))

    def _dispatch(self, fn: Callable[..., None], *args: object) -> None:
        if not self._app.is_running:
            return
        with contextlib.suppress(RuntimeError):
            self._app.call_from_thread(fn, *args)


class LiveAgentApp(App[None]):
    """The hands-free voice TUI: a scrolling transcript above an animated voice bar."""

    # Flat pure-black canvas matching the `code` TUI: a bordered voice bar and a one-line
    # footer, with no text prompt (there's nothing to type into a live voice session).
    CSS = f"""
    Screen {{ background: #000000; }}
    #log {{ height: 1fr; border: none; background: #000000; padding: 1 2; }}
    #voicebar {{ dock: bottom; height: 3; background: #000000; border: round {banner.BRAND_HEX};
        margin: 1 1; content-align: center middle; }}
    #status {{ dock: bottom; height: 1; background: #000000; padding: 0 1; }}
    /* Blank line above each agent reply (and the greeting), so turns don't run together. */
    AssistantMessage {{ margin-top: 1; }}
    """
    TITLE = "AssemblyAI Live"
    ENABLE_COMMAND_PALETTE = False
    # Escape and Ctrl-C interrupt a playing reply (silence it and drop back to listening),
    # the same as talking over the agent — so you can stop a long answer without speaking.
    # When nothing is speaking, Ctrl-C quits; Ctrl-Q always quits (the guaranteed escape
    # hatch, so a stuck reply can never trap the session). Quitting closes the audio, which
    # unblocks the cascade worker.
    BINDINGS: ClassVar = [
        ("escape", "interrupt", "Interrupt"),
        ("ctrl+c", "interrupt_or_quit", "Interrupt / Quit"),
        ("ctrl+q", "stop", "Quit"),
    ]

    def __init__(
        self,
        *,
        run_conversation: Callable[[Renderer], None],
        on_stop: Callable[[], None],
        web_note: str | None = None,
    ) -> None:
        super().__init__()
        self._run_conversation = run_conversation  # blocking; runs the cascade given a Renderer
        self._on_stop = on_stop  # closes the audio so a quit unblocks the cascade worker
        self._web_note = web_note
        # The cascade's reply-interrupt, wired once its session exists (see set_interrupt);
        # None until then, so an early keypress is a harmless no-op.
        self._interrupt: Callable[[], bool] | None = None
        self._voice_phase = "listening"
        self._voice_frames = itertools.cycle(tui_status.VOICE_FRAMES)
        self._voice_timer: Timer | None = None
        self._user_partial: UserMessage | None = None  # the in-place "you: …" widget for a turn
        self._reply_msg: AssistantMessage | None = None  # the reply widget sentences stream into
        self._stopped = False  # guards on_stop against a double teardown (quit + unmount)

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="log")
        yield Static("", id="voicebar")
        yield Static(f"[dim]{_STATUS_LINE}[/dim]", id="status")

    def on_mount(self) -> None:
        self._write_splash()
        if self._web_note:
            self.notify(self._web_note, title="Web search disabled", severity="warning")
        self._render_voicebar()
        self._voice_timer = self.set_interval(0.3, self._tick_voice)  # pragma: no mutate
        # Defer the first mic open until after the splash has painted (a GIL-holding PortAudio
        # open races Textual's initial render otherwise — same reason as the code TUI).
        self.call_after_refresh(self._start)

    def _start(self) -> None:
        # thread=True: the cascade is a blocking sync call; exclusive=True: one session at a time.
        self.run_worker(self._run, thread=True, exclusive=True, name="cascade")  # pragma: no mutate

    def _run(self) -> None:
        """Drive the cascade on a worker thread, then close the app when it ends."""
        renderer = _TuiRenderer(self)
        try:
            self._run_conversation(renderer)
        except CLIError as exc:
            self._safely(self._show_error, exc.message)
        # The cascade returned (STT closed, a leg failed, or a quit closed the audio) — exit.
        self._safely(self.exit)

    def _safely(self, fn: Callable[..., None], *args: object) -> None:
        """Hop ``fn`` onto the UI thread, dropping the error a torn-down app raises mid-call."""
        if not self.is_running:
            return
        with contextlib.suppress(RuntimeError):
            self.call_from_thread(fn, *args)

    # --- transcript (always called on the UI thread) --------------------------

    def live_connected(self) -> None:
        """The session is live; the splash already shows the listening prompt."""
        self._set_phase("listening")

    def show_user_partial(self, text: str) -> None:
        """Grow the interim user transcript in place while the turn is still being spoken."""
        self._set_phase("listening")
        if self._user_partial is None:
            self._user_partial = UserMessage(text)
            self._mount(self._user_partial)
        else:
            self._user_partial.set_text(text)
            self._scroll_end()

    def show_user_final(self, text: str) -> None:
        """Commit the finalized user turn and move to the thinking phase."""
        if self._user_partial is None:
            self._mount(UserMessage(text))
        else:
            self._user_partial.set_text(text)
        self._user_partial = None  # finalized; the next partial starts a fresh line
        self._set_phase("thinking")
        self._scroll_end()

    def begin_reply(self) -> None:
        """Open a fresh reply widget the agent's sentences stream into; switch to speaking."""
        self._set_phase("speaking")
        self._reply_msg = AssistantMessage()
        self._mount(self._reply_msg)

    def show_agent_sentence(self, text: str) -> None:
        """Append one spoken sentence to the in-flight reply."""
        if self._reply_msg is None:
            self._reply_msg = AssistantMessage()
            self._mount(self._reply_msg)
        self._reply_msg.stream(f"{text} ")
        self._scroll_end()

    def end_reply(self, *, interrupted: bool) -> None:
        """Finalize the reply (rendered as Markdown) and return to listening."""
        if self._reply_msg is not None:
            self._reply_msg.finalize(self._reply_msg.text)
            self._reply_msg = None
        if interrupted:
            self._mount(Note("(interrupted)"))
        self._set_phase("listening")

    def _show_error(self, message: str) -> None:
        self._mount(ErrorMessage(message))

    # --- voice bar ------------------------------------------------------------

    def _set_phase(self, phase: str) -> None:
        self._voice_phase = phase
        self._render_voicebar()

    def _render_voicebar(self) -> None:
        """Paint the voice bar for the current phase (no Ctrl-V hint — input is voice-only)."""
        self.query_one("#voicebar", Static).update(
            tui_status.voicebar_markup(self._voice_phase, next(self._voice_frames))
        )

    def _tick_voice(self) -> None:
        """Advance the voice-bar meter one frame (the animation timer's callback)."""
        self._render_voicebar()

    # --- splash / mounting ----------------------------------------------------

    def _write_splash(self) -> None:
        rows = [f"[bold {banner.BRAND_HEX}]{row}[/]" for row in banner.wordmark()]
        rows += [
            f"[dim]{banner.version()}[/dim]",
            "",
            f"[{banner.BRAND_HEX}]{_READY_LINE}[/]",
            f"[dim]{_TIP_LINE}[/dim]",
        ]
        self._mount(Static("\n".join(rows)))

    def _mount(self, widget: Static) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)  # pragma: no mutate — cosmetic; animate flag is unassertable

    def _scroll_end(self) -> None:
        self.query_one("#log", VerticalScroll).scroll_end(animate=False)  # pragma: no mutate

    # --- interrupt / quit -----------------------------------------------------

    def set_interrupt(self, interrupt: Callable[[], bool]) -> None:
        """Wire the session's reply-interrupt once the cascade has built its session.

        Called from the cascade worker thread (via ``run_cascade``'s ``on_session``); it only
        stores a callable reference, so no UI hop is needed.
        """
        self._interrupt = interrupt

    def action_interrupt(self) -> None:
        """Escape: silence a playing reply and return to listening (a no-op when idle)."""
        self._do_interrupt()

    def action_interrupt_or_quit(self) -> None:
        """Ctrl-C: silence a playing reply and keep listening; quit when nothing is speaking."""
        if not self._do_interrupt():
            self.action_stop()

    def _do_interrupt(self) -> bool:
        """Fire the session's reply-interrupt; True if a reply was playing.

        The reply worker then unwinds and emits ``reply_done``, so the renderer is what
        returns the voice bar to listening — this only has to signal the stop.
        """
        return self._interrupt is not None and self._interrupt()

    def action_stop(self) -> None:
        """Ctrl-Q (or Ctrl-C when idle): stop the audio (unblocking the worker) and exit."""
        self._teardown()
        self.exit()

    def on_unmount(self) -> None:
        """Close the audio on any exit path, in case the worker is still blocked on the mic."""
        self._teardown()

    def _teardown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._on_stop()
