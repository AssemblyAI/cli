"""Surface control-loop progress: human lines on stderr, the reply on stdout.

In human mode the running narration (what was heard, each action, refusals) goes
to the stderr console so stdout carries only the agent's spoken reply — the
pipe-safe split the rest of the CLI keeps. In ``--json`` mode every event is one
NDJSON record on stdout, each tagged with a ``type`` field per the streaming
output convention.
"""

from __future__ import annotations

from aai_cli.control.actions import Action
from aai_cli.ui import output


def _describe(action: Action) -> str:
    """A compact one-line description of an action and its arguments."""
    if action.arguments:
        return f"{action.name} {action.arguments}"
    return action.name


class ControlRenderer:
    """Render engine events for one ``assembly control`` session."""

    def __init__(self, *, json_mode: bool) -> None:
        self._json = json_mode

    def _event(self, event_type: str, **fields: object) -> None:
        output.emit_ndjson({"type": event_type, **fields})

    def on_user(self, text: str) -> None:
        """A finalized spoken instruction was heard."""
        if self._json:
            self._event("user", text=text)
        else:
            output.error_console.print(output.muted(f"you: {text}"))

    def on_action(self, action: Action) -> None:
        """An action is about to run on the host."""
        if self._json:
            self._event("action", action=action.name, arguments=action.arguments)
        else:
            output.error_console.print(output.muted(f"→ {_describe(action)}"))

    def on_result(self, action: Action, result: dict[str, object]) -> None:
        """An action finished, with the helper's result."""
        if self._json:
            self._event("result", action=action.name, result=result)
        elif result.get("ok") is False:
            output.error_console.print(output.warn(f"  {result.get('error', 'failed')}"))

    def on_refused(self, action: Action, reason: str) -> None:
        """A UI-mutating action was refused (e.g. ``--dry-run``)."""
        if self._json:
            self._event("refused", action=action.name, reason=reason)
        else:
            output.error_console.print(output.warn(f"refused {action.name}: {reason}"))

    def on_invalid(self, reason: str) -> None:
        """The model called an unknown/under-specified tool."""
        if self._json:
            self._event("invalid", reason=reason)
        else:
            output.error_console.print(output.warn(reason))

    def on_reply(self, text: str) -> None:
        """The model's spoken reply that ends a turn."""
        if self._json:
            self._event("reply", text=text)
        else:
            output.console.print(text)
