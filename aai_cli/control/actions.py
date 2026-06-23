"""The action protocol: the vocabulary the LLM "brain" uses to drive the macOS
"hands" helper.

An :class:`Action` is one tool call the model emitted — a name plus JSON
arguments. :func:`validate` checks the name is known and the required arguments
are present, turning a raw model tool call into a request the Swift helper
understands. Everything here is pure data, so the engine is exercised without a
model, a microphone, or macOS.
"""

from __future__ import annotations

from dataclasses import dataclass

# Action name -> the argument names it requires. The Swift helper understands
# exactly these actions; a tool call for any other name is rejected back to the
# model and never executed (see :func:`validate`).
ACTION_SPECS: dict[str, tuple[str, ...]] = {
    "type_text": ("text",),
    "key_combo": ("keys",),
    "click": (),
    "launch_app": ("name",),
    "focus_app": ("name",),
    "get_ui_tree": (),
    "screenshot": (),
}

# Actions that only read the screen and never change UI state. `--dry-run`
# executes these for real (so the model can still "see") but refuses every
# other, UI-mutating action.
OBSERVE_ACTIONS = frozenset({"get_ui_tree", "screenshot"})


class InvalidAction(Exception):
    """A model tool call that names an unknown action or omits a required argument.

    Surfaced back to the model as a failed tool result rather than crashing the
    session — the model can correct itself on the next step.
    """


@dataclass(frozen=True)
class Action:
    """One validated UI action: a known name plus its JSON arguments."""

    name: str
    arguments: dict[str, object]

    def is_observe(self) -> bool:
        """True for read-only actions (screen observation), which `--dry-run` allows."""
        return self.name in OBSERVE_ACTIONS

    def request(self) -> dict[str, object]:
        """The JSON object sent to the Swift helper: the action name plus its arguments."""
        return {"action": self.name, **self.arguments}


def validate(name: str, arguments: dict[str, object]) -> Action:
    """Turn a model's tool call into an :class:`Action`, or raise :class:`InvalidAction`.

    Rejects an unknown action name and any call missing a required argument, so the
    helper is only ever handed a request it can execute.
    """
    required = ACTION_SPECS.get(name)
    if required is None:
        raise InvalidAction(f"Unknown action {name!r}.")
    missing = [arg for arg in required if arg not in arguments]
    if missing:
        raise InvalidAction(
            f"Action {name!r} is missing required argument(s): {', '.join(missing)}."
        )
    return Action(name=name, arguments=arguments)
